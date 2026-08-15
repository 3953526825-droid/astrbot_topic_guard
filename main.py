# -*- coding: utf-8 -*-
"""
AstrBot 插件：话题守卫（topic_guard）
自动检测对话话题是否结束并主动停止对话，支持双机器人互聊与人机对聊。
"""
import asyncio
import difflib
import json
import os
import re
import time
from collections import deque

from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star

try:
    from astrbot.api import AstrBotConfig
except ImportError:  # 极老版本兜底
    AstrBotConfig = dict

try:
    from astrbot.api.event import MessageChain
except ImportError:  # 旧版本兜底
    from astrbot.api.message_components import MessageChain

DEFAULT_JUDGE_SYSTEM_PROMPT = (
    "你是一个「对话话题状态检测器」。你的任务：判断给定的一段对话是否已经自然聊完、可以收尾停止。\n"
    "判定为「已结束」的典型情形：\n"
    "1. 双方在互相告别或客套收尾，例如“好的再见”“下次聊”“谢谢，没别的问题了”。\n"
    "2. 对方提出的问题已经被解决，并且没有再提出新问题或新需求。\n"
    "3. 对话陷入敷衍、重复，或只剩客套（如连续“嗯嗯”“好的”“哈哈”）。\n"
    "4. 对方明确表示要离开、下线、去忙别的。\n"
    "判定为「未结束」的典型情形：\n"
    "1. 对方提出了新问题、新话题，或明显还想继续聊。\n"
    "2. 对方正在表达情绪、观点或展开叙述。\n"
    "3. 对话中仍有未完成的任务。\n"
    "注意：宁可保守一点，没有把握时判为未结束。\n"
    "你必须只输出一个 JSON 对象，不要输出任何其他文字，格式："
    '{"ended": true 或 false, "score": 0到1的小数, "reason": "一句话理由"}'
)

DEFAULT_SUMMARY_PROMPT = "你是一个对话总结助手。请用一到两句话（不超过60字）总结这段对话的主要内容，语气自然。"

DEFAULT_END_KEYWORDS = ["再见", "拜拜", "下次聊", "下次再聊", "改天聊", "回头聊", "先这样", "就到这", "不聊了", "下了", "告辞", "晚安", "886", "goodbye", "bye", "拜"]

DEFAULT_RESUME_KEYWORDS = ["在吗", "在不在", "在么", "新话题", "问个问题", "帮我", "你好", "hello", "hi", "呼叫"]

DEFAULT_CLOSING = "🌙 这个话题聊得差不多啦，我先去待机了～有新话题随时喊我！"

DEFAULTS = {
    "enable": True,
    "llm_judge_enable": True,
    "judge_model": "",
    "judge_threshold": 0.7,
    "judge_consecutive": 2,
    "judge_trigger_max_len": 40,
    "judge_interval": 3,
    "judge_bot_reply": True,
    "judge_bot_reply_max_len": 60,
    "judge_system_prompt": DEFAULT_JUDGE_SYSTEM_PROMPT,
    "end_keywords": DEFAULT_END_KEYWORDS,
    "silence_enable": True,
    "silence_minutes": 10,
    "silence_check_interval": 30,
    "repeat_enable": True,
    "repeat_similarity": 0.85,
    "repeat_max": 3,
    "max_turns": 30,
    "min_turns_before_stop": 2,
    "closing_mode": "message",
    "closing_message": DEFAULT_CLOSING,
    "closing_cooldown": 300,
    "summary_prompt": DEFAULT_SUMMARY_PROMPT,
    "resume_notice": "",
    "auto_resume": True,
    "resume_min_len": 8,
    "resume_question_trigger": True,
    "resume_keywords": DEFAULT_RESUME_KEYWORDS,
    "resume_at_me": True,
    "resume_cooldown_secs": 20,
    "stop_ttl_minutes": 60,
    "other_bot_ids": [],
    "track_all_senders": False,
    "engage_window_minutes": 10,
    "group_whitelist": [],
    "group_blacklist": [],
    "user_blacklist": [],
    "history_max": 24,
    "session_ttl_days": 3,
    "state_file": "data/topic_guard_state.json",
    "debug_log": False,
}


class TopicGuard(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.cfg = config if isinstance(config, dict) else {}
        self.sessions = {}          # 会话ID -> 会话状态
        self._stats = {"total_stops": 0, "reasons": {}}
        self._dirty = False
        self._sweeper_task = None
        self._sweeper_started = False
        self._load_state()

    # ---------------- 小工具 ----------------
    def _c(self, key):
        v = self.cfg.get(key) if isinstance(self.cfg, dict) else None
        return v if v is not None else DEFAULTS.get(key)

    def _enabled(self):
        try:
            return bool(self._c("enable"))
        except Exception:
            return True

    def _conv_id(self, event):
        gid = event.get_group_id()
        if gid:
            return f"g:{gid}"
        return f"p:{event.get_sender_id()}"

    def _is_self(self, event):
        """忽略机器人自己的消息回显"""
        try:
            mo = event.message_obj
            if mo is None:
                return False
            self_id = getattr(mo, "self_id", None)
            if self_id and str(self_id) == str(event.get_sender_id()):
                return True
        except Exception:
            pass
        return False

    def _is_at_me(self, event):
        try:
            mo = event.message_obj
            if mo is None:
                return False
            self_id = str(getattr(mo, "self_id", "") or "")
            if not self_id:
                return False
            for comp in (getattr(mo, "message", None) or []):
                qq = getattr(comp, "qq", None)
                if qq is not None and str(qq) == self_id:
                    return True
        except Exception:
            pass
        return False

    def _session_skip(self, event):
        cid = self._conv_id(event)
        bl = [str(x).strip() for x in self._c("group_blacklist") if str(x).strip()]
        if cid in bl:
            return True
        wl = [str(x).strip() for x in self._c("group_whitelist") if str(x).strip()]
        if wl and cid not in wl:
            return True
        ubl = [str(x).strip() for x in self._c("user_blacklist") if str(x).strip()]
        if str(event.get_sender_id() or "") in ubl:
            return True
        return False

    def _is_counterpart(self, sender_id):
        """判断发送者是否属于对话对方（双机器人场景按配置过滤）"""
        ids = [str(x).strip() for x in self._c("other_bot_ids") if str(x).strip()]
        if not ids:
            return True
        if sender_id in ids:
            return True
        return bool(self._c("track_all_senders"))

    def _new_session(self, cid, umo=None):
        return {
            "cid": cid,
            "umo": umo,
            "status": "active",          # active | stopped
            "history": deque(maxlen=max(int(self._c("history_max")), 2)),
            "turns": 0,                  # 本轮话题对方消息数
            "total_turns": 0,
            "last_peer_ts": 0.0,
            "last_bot_ts": 0.0,
            "end_ema": 0.0,              # 话题结束倾向 EMA
            "consecutive_end": 0,
            "judged_since": 0,
            "repeat_buf": deque(maxlen=max(int(self._c("repeat_max")), 2)),
            "stopped_at": 0.0,
            "stop_reason": "",
            "closing_sent_at": 0.0,
            "peer_id": "",
            "peer_name": "",
            "judging": False,
        }

    def _ensure_session(self, event):
        cid = self._conv_id(event)
        if cid not in self.sessions:
            self.sessions[cid] = self._new_session(cid, event.unified_msg_origin)
        s = self.sessions[cid]
        if not s.get("umo"):
            try:
                s["umo"] = event.unified_msg_origin
            except Exception:
                pass
        return s

    def _mark_dirty(self):
        self._dirty = True

    def _hit_end_keywords(self, text):
        if not text:
            return False
        low = text.lower()
        for kw in self._c("end_keywords"):
            if kw and kw.lower() in low:
                return True
        return False

    def _record_peer_msg(self, session, event, text):
        session["peer_id"] = str(event.get_sender_id() or "")
        session["peer_name"] = (event.get_sender_name() or "")[:30]
        session["last_peer_ts"] = time.time()
        session["history"].append({"role": "peer", "text": text[:800], "ts": time.time()})
        session["turns"] += 1
        session["total_turns"] += 1
        self._mark_dirty()

    # ---------------- 群聊防干扰：是否计入对话 ----------------
    def _should_track_peer_msg(self, event, session, text):
        if not event.get_group_id():
            return True                                    # 私聊必计
        if self._is_at_me(event):
            return True                                    # 明确在和机器人说话
        sender_id = str(event.get_sender_id() or "")
        if sender_id in [str(x).strip() for x in self._c("other_bot_ids") if str(x).strip()]:
            return True                                    # 对方机器人消息必计
        if self._c("track_all_senders"):
            return True
        window = float(self._c("engage_window_minutes")) * 60
        return bool(session["last_bot_ts"]) and (time.time() - session["last_bot_ts"]) <= window

    # ---------------- 新话题恢复 ----------------
    def _should_resume(self, event, session, text, is_group):
        if not self._c("auto_resume"):
            return False
        if session.get("stopped_at") and (time.time() - session["stopped_at"]) < float(self._c("resume_cooldown_secs")):
            return False
        if self._c("resume_at_me") and self._is_at_me(event):
            return True
        if self._c("resume_question_trigger") and ("?" in text or "？" in text):
            return True
        for kw in self._c("resume_keywords"):
            if kw and kw in text:
                return True
        # 私聊里较长消息视为新话题；群聊为避免误唤醒不按长度恢复
        if not is_group and len(text) >= int(self._c("resume_min_len")):
            return True
        return False

    # ---------------- 重复内容检测 ----------------
    def _check_repeat(self, session, text):
        if not self._c("repeat_enable"):
            return False
        buf = session["repeat_buf"]
        buf.append(text)
        need = int(self._c("repeat_max"))
        if len(buf) < max(need, 2):
            return False
        items = list(buf)
        sim = float(self._c("repeat_similarity"))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                try:
                    r = difflib.SequenceMatcher(None, items[i], items[j]).ratio()
                except Exception:
                    continue
                if r < sim:
                    return False
        return True

    # ---------------- LLM 判定 ----------------
    def _format_history(self, session, extra=None):
        maxn = int(self._c("history_max"))
        lines = []
        for h in list(session["history"])[-maxn:]:
            who = "对方" if h["role"] == "peer" else "本机器人"
            lines.append(f"{who}：{str(h['text'])[:200]}")
        if extra:
            lines.append(f"本机器人：{str(extra)[:200]}")
        return "\n".join(lines) or "（暂无对话记录）"

    @staticmethod
    def _extract_json(text):
        if not text:
            return None
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S).strip()
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            text = m.group(0)
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        ended_m = re.search(r'"ended"\s*:\s*(true|false)', text, re.I)
        score_m = re.search(r'"score"\s*:\s*([0-9.]+)', text)
        if ended_m:
            return {
                "ended": ended_m.group(1).lower() == "true",
                "score": float(score_m.group(1)) if score_m else (1.0 if ended_m.group(1).lower() == "true" else 0.0),
            }
        return None

    @staticmethod
    def _llm_response_text(resp):
        if resp is None:
            return ""
        t = getattr(resp, "completion_text", "") or ""
        if t and str(t).strip():
            return str(t).strip()
        chain = getattr(resp, "result_chain", None)
        if chain is not None:
            try:
                pt = chain.get_plain_text()
                if pt:
                    return str(pt).strip()
            except Exception:
                pass
            try:
                parts = []
                for c in chain:
                    txt = getattr(c, "text", None)
                    if isinstance(txt, str):
                        parts.append(txt)
                return "".join(parts).strip()
            except Exception:
                pass
        return ""

    async def _get_provider(self, session):
        ctx = self.context
        umo = session.get("umo") or ""
        if hasattr(ctx, "get_using_provider_async"):
            try:
                return await ctx.get_using_provider_async(umo=umo)
            except Exception as e:
                logger.warning(f"[topic_guard] get_using_provider_async 失败: {e}")
        if hasattr(ctx, "get_using_provider"):
            try:
                return ctx.get_using_provider()
            except Exception as e:
                logger.warning(f"[topic_guard] get_using_provider 失败: {e}")
        return None

    async def _llm_judge(self, session, reason=""):
        """后台任务：让 LLM 判定话题是否结束"""
        if session.get("judging") or session["status"] != "active":
            return
        session["judging"] = True
        try:
            provider = await self._get_provider(session)
            if provider is None:
                logger.info("[topic_guard] 当前会话无可用 LLM Provider，跳过智能判定（仍使用沉默/重复/轮次规则）")
                return
            history_text = self._format_history(session)
            prompt = (
                "请判断下面这段对话的话题是否已经自然结束。\n\n"
                f"【对话记录】\n{history_text}\n\n"
                "【任务】判断“对方”与“本机器人”的这段对话是否已经聊完、可以收尾停止。"
                '只输出 JSON：{"ended": true 或 false, "score": 0到1的小数, "reason": "一句话理由"}'
            )
            kwargs = {"prompt": prompt, "system_prompt": str(self._c("judge_system_prompt"))}
            model = str(self._c("judge_model")).strip()
            if model:
                kwargs["model"] = model
            resp = await provider.text_chat(**kwargs)
            raw = self._llm_response_text(resp)
            data = self._extract_json(raw)
            session["judged_since"] = session["turns"]
            if data is None:
                logger.warning(f"[topic_guard] 判定输出无法解析: {raw[:150] if raw else ''}")
                return
            ended = bool(data.get("ended"))
            try:
                score = float(data.get("score", 1.0 if ended else 0.0))
            except Exception:
                score = 1.0 if ended else 0.0
            score = max(0.0, min(1.0, score))
            session["end_ema"] = session["end_ema"] * 0.6 + score * 0.4
            if ended:
                session["consecutive_end"] += 1
            else:
                session["consecutive_end"] = 0
            if self._c("debug_log"):
                logger.info(f"[topic_guard] 判定 {session['cid']}: ended={ended} score={score:.2f} "
                            f"ema={session['end_ema']:.2f} 连续={session['consecutive_end']} ({reason})")
            if (session["status"] == "active"
                    and session["end_ema"] >= float(self._c("judge_threshold"))
                    and session["consecutive_end"] >= int(self._c("judge_consecutive"))):
                await self._do_stop(session, f"LLM判定：话题结束（{reason}）")
        except Exception as e:
            logger.error(f"[topic_guard] LLM 判定异常: {e}")
        finally:
            session["judging"] = False

    # ---------------- 停止 / 恢复 / 发送 ----------------
    async def _do_stop(self, session, reason, manual=False):
        if session["status"] == "stopped":
            return
        session["status"] = "stopped"
        session["stopped_at"] = time.time()
        session["stop_reason"] = reason
        session["consecutive_end"] = 0
        self._stats["total_stops"] = self._stats.get("total_stops", 0) + 1
        self._stats.setdefault("reasons", {})
        self._stats["reasons"][reason] = self._stats["reasons"].get(reason, 0) + 1
        self._save_state()
        logger.info(f"[topic_guard] 会话 {session['cid']} 已停止: {reason}")
        if not manual:
            await self._send_closing(session)

    async def _do_resume(self, session, reason):
        was_stopped = session["status"] == "stopped"
        session["status"] = "active"
        session["end_ema"] = 0.0
        session["consecutive_end"] = 0
        session["turns"] = 0
        session["judged_since"] = 0
        session["repeat_buf"].clear()
        session["history"].clear()
        session["stop_reason"] = ""
        if was_stopped:
            logger.info(f"[topic_guard] 会话 {session['cid']} 已恢复: {reason}")
            notice = str(self._c("resume_notice")).strip()
            if notice:
                await self._safe_send(session, notice)
        self._save_state()

    async def _send_closing(self, session):
        mode = str(self._c("closing_mode"))
        if mode == "silent":
            return
        umo = session.get("umo")
        if not umo:
            return
        now = time.time()
        if now - session.get("closing_sent_at", 0) < float(self._c("closing_cooldown")):
            return
        session["closing_sent_at"] = now
        text = str(self._c("closing_message"))
        if mode == "summary":
            summary = await self._summarize(session)
            if summary:
                text = f"{summary}\n\n{text}"
        await self._safe_send(session, text)

    async def _summarize(self, session):
        try:
            provider = await self._get_provider(session)
            if provider is None:
                return None
            history_text = self._format_history(session)
            resp = await provider.text_chat(
                prompt=f"请总结以下对话：\n\n{history_text}",
                system_prompt=str(self._c("summary_prompt")),
            )
            text = self._llm_response_text(resp)
            return text[:300] if text else None
        except Exception as e:
            logger.warning(f"[topic_guard] 生成总结失败: {e}")
            return None

    async def _safe_send(self, session, text):
        try:
            if not hasattr(self.context, "send_message"):
                logger.warning("[topic_guard] 当前 AstrBot 版本不支持 context.send_message")
                return
            await self.context.send_message(session["umo"], MessageChain().message(text))
        except Exception as e:
            logger.warning(f"[topic_guard] 发送消息失败: {e}")

    # ---------------- 持久化 ----------------
    def _state_path(self):
        return os.path.abspath(str(self._c("state_file")))

    def _save_state(self):
        try:
            path = self._state_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            sessions_out = {}
            for cid, s in self.sessions.items():
                sessions_out[cid] = {
                    "umo": s.get("umo"),
                    "status": s["status"],
                    "stopped_at": s.get("stopped_at", 0),
                    "stop_reason": s.get("stop_reason", ""),
                    "last_peer_ts": s.get("last_peer_ts", 0),
                    "last_bot_ts": s.get("last_bot_ts", 0),
                    "turns": s.get("turns", 0),
                    "total_turns": s.get("total_turns", 0),
                    "peer_name": s.get("peer_name", ""),
                }
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"sessions": sessions_out, "stats": self._stats}, f, ensure_ascii=False, indent=2)
            self._dirty = False
        except Exception as e:
            logger.warning(f"[topic_guard] 保存状态失败: {e}")

    def _load_state(self):
        try:
            path = self._state_path()
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cid, s in (data.get("sessions") or {}).items():
                sess = self._new_session(cid, s.get("umo"))
                for k in ("status", "stopped_at", "stop_reason", "last_peer_ts", "last_bot_ts", "turns", "total_turns", "peer_name"):
                    if k in s:
                        sess[k] = s[k]
                self.sessions[cid] = sess
            if isinstance(data.get("stats"), dict):
                self._stats.update(data["stats"])
            logger.info(f"[topic_guard] 已恢复 {len(self.sessions)} 个会话状态")
        except Exception as e:
            logger.warning(f"[topic_guard] 读取状态失败: {e}")

    # ---------------- 后台巡检 ----------------
    def _ensure_sweeper(self):
        if self._sweeper_started:
            return
        self._sweeper_started = True
        self._sweeper_task = asyncio.create_task(self._sweep_loop())

    async def _sweep_loop(self):
        while True:
            try:
                await asyncio.sleep(max(5, int(self._c("silence_check_interval"))))
                if not self._enabled():
                    continue
                if self._dirty:
                    self._save_state()
                now = time.time()
                if self._c("silence_enable"):
                    silence = float(self._c("silence_minutes")) * 60
                    min_turns = int(self._c("min_turns_before_stop"))
                    for s in list(self.sessions.values()):
                        if s["status"] != "active" or s["turns"] < min_turns:
                            continue
                        last_act = max(s.get("last_peer_ts", 0), s.get("last_bot_ts", 0))
                        if last_act and (now - last_act) >= silence:
                            await self._do_stop(s, f"沉默超时 {int((now - last_act) // 60)} 分钟")
                ttl = int(self._c("stop_ttl_minutes"))
                if ttl > 0:
                    for s in list(self.sessions.values()):
                        if s["status"] == "stopped" and s.get("stopped_at") and (now - s["stopped_at"]) >= ttl * 60:
                            await self._do_resume(s, "TTL 到期自动恢复")
                days = int(self._c("session_ttl_days"))
                if days > 0:
                    cutoff = now - days * 86400
                    for cid, s in list(self.sessions.items()):
                        last = max(s.get("last_peer_ts", 0), s.get("last_bot_ts", 0))
                        if last and last < cutoff:
                            self.sessions.pop(cid, None)
                            self._dirty = True
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[topic_guard] 后台任务异常: {e}")

    # ---------------- 事件处理 ----------------
    async def _handle_message(self, event):
        try:
            if not self._enabled() or self._is_self(event) or self._session_skip(event):
                return
            self._ensure_sweeper()
            session = self._ensure_session(event)
            text = (event.message_str or "").strip()
            is_group = bool(event.get_group_id())

            # 已停止：尝试检测新话题
            if session["status"] == "stopped":
                if self._should_resume(event, session, text, is_group):
                    await self._do_resume(session, "检测到新话题")
                    self._record_peer_msg(session, event, text)
                return

            if not self._is_counterpart(str(event.get_sender_id() or "")):
                return
            if not self._should_track_peer_msg(event, session, text):
                return

            self._record_peer_msg(session, event, text)

            # 1) 最大轮次保护
            max_turns = int(self._c("max_turns"))
            if max_turns > 0 and session["turns"] >= max_turns:
                await self._do_stop(session, f"达到最大轮次({max_turns})")
                return

            # 2) 车轱辘话快速通道
            if self._check_repeat(session, text):
                await self._do_stop(session, "连续重复内容（疑似车轱辘话）")
                return

            # 3) LLM 判定触发条件
            if (self._c("llm_judge_enable")
                    and session["turns"] >= int(self._c("min_turns_before_stop"))
                    and not session["judging"]):
                kw_hit = self._hit_end_keywords(text)
                short = len(text) <= int(self._c("judge_trigger_max_len"))
                interval_hit = (session["turns"] - session["judged_since"]) >= int(self._c("judge_interval"))
                if kw_hit or short or interval_hit:
                    reason = "命中结束关键词" if kw_hit else ("对方短消息" if short else "轮次间隔")
                    asyncio.create_task(self._llm_judge(session, reason))
        except Exception as e:
            logger.error(f"[topic_guard] 消息处理异常: {e}")

    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        await self._handle_message(event)

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE)
    async def on_private_message(self, event: AstrMessageEvent):
        await self._handle_message(event)

    # 话题结束后拦截 LLM 回复，让机器人保持静默
    @filter.on_llm_request()
    async def hook_on_llm_request(self, event: AstrMessageEvent, _req):
        try:
            if not self._enabled() or self._is_self(event):
                return
            session = self.sessions.get(self._conv_id(event))
            if session and session["status"] == "stopped":
                event.stop_event()
        except Exception as e:
            logger.error(f"[topic_guard] on_llm_request 异常: {e}")

    # 记录机器人自己的回复，并判断是否在收尾
    @filter.on_llm_response()
    async def hook_on_llm_response(self, event: AstrMessageEvent, resp):
        try:
            if not self._enabled() or self._is_self(event):
                return
            session = self.sessions.get(self._conv_id(event))
            if not session or session["status"] != "active":
                return
            text = self._llm_response_text(resp)
            if not text:
                return
            session["history"].append({"role": "bot", "text": text[:800], "ts": time.time()})
            session["last_bot_ts"] = time.time()
            self._mark_dirty()
            if not (self._c("judge_bot_reply") and self._c("llm_judge_enable")):
                return
            if session["turns"] < int(self._c("min_turns_before_stop")):
                return
            if len(text) > int(self._c("judge_bot_reply_max_len")):
                return
            kw_hit = self._hit_end_keywords(text)
            interval_hit = (session["turns"] - session["judged_since"]) >= int(self._c("judge_interval"))
            if (kw_hit or interval_hit) and not session["judging"]:
                asyncio.create_task(self._llm_judge(session, "机器人回复疑似收尾" if kw_hit else "机器人短回复"))
        except Exception as e:
            logger.error(f"[topic_guard] on_llm_response 异常: {e}")

    # ---------------- 指令 ----------------
    @filter.command("tstop", alias={"话题停止", "结束对话"})
    async def cmd_stop(self, event: AstrMessageEvent):
        '''手动结束当前话题，机器人将保持静默，直到检测到新话题。'''
        session = self._ensure_session(event)
        if session["status"] == "stopped":
            yield event.plain_result("🤖 本会话已处于停止状态。发送 /tresume 或提出新话题即可恢复。")
            event.stop_event()
            return
        await self._do_stop(session, "手动指令停止", manual=True)
        yield event.plain_result("✅ 已结束当前话题，我将保持静默。恢复方式：新话题 / @我 / /tresume")
        event.stop_event()

    @filter.command("tresume", alias={"话题恢复", "继续对话"})
    async def cmd_resume(self, event: AstrMessageEvent):
        '''恢复本会话对话。'''
        session = self._ensure_session(event)
        await self._do_resume(session, "手动指令恢复")
        yield event.plain_result("✅ 已恢复对话，话题状态与历史已重置。")
        event.stop_event()

    @filter.command("treset", alias={"话题重置"})
    async def cmd_reset(self, event: AstrMessageEvent):
        '''重置本会话的话题状态与历史记录。'''
        session = self._ensure_session(event)
        await self._do_resume(session, "手动指令重置")
        yield event.plain_result("✅ 已重置本会话的话题状态与历史记录。")
        event.stop_event()

    @filter.command("tstatus", alias={"话题状态"})
    async def cmd_status(self, event: AstrMessageEvent):
        '''查看当前会话的话题状态。'''
        session = self._ensure_session(event)
        now = time.time()
        if session["status"] == "stopped":
            state = "🔴 已停止"
            detail = f"原因：{session.get('stop_reason') or '未知'}"
            if session.get("stopped_at"):
                detail += f"\n已停止 {int((now - session['stopped_at']) // 60)} 分钟"
            ttl = int(self._c("stop_ttl_minutes"))
            detail += "\n恢复：提出新话题 / @我 / /tresume" + (f"\nTTL：{ttl} 分钟后自动恢复" if ttl > 0 else "")
        else:
            state = "🟢 进行中"
            last_act = max(session.get("last_peer_ts", 0), session.get("last_bot_ts", 0))
            idle = int(now - last_act) if last_act else 0
            detail = f"本轮对方消息：{session.get('turns', 0)} 条（累计 {session.get('total_turns', 0)} 条）"
            detail += f"\n结束倾向 EMA：{session.get('end_ema', 0):.2f}（阈值 {float(self._c('judge_threshold'))}）"
            detail += f"\n已静默：{idle // 60} 分 {idle % 60} 秒"
        yield event.plain_result(f"🗨️ 话题守卫 · 本会话状态\n{state}\n{detail}\n\n发送 /tguard 查看全部指令")
        event.stop_event()

    @filter.command("tstat", alias={"话题统计"})
    async def cmd_stat(self, event: AstrMessageEvent):
        '''查看话题守卫统计信息。'''
        active = sum(1 for s in self.sessions.values() if s["status"] == "active")
        stopped = sum(1 for s in self.sessions.values() if s["status"] == "stopped")
        total_turns = sum(s.get("total_turns", 0) for s in self.sessions.values())
        reasons = self._stats.get("reasons") or {}
        reason_str = ", ".join(f"{k}×{v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1])) or "暂无"
        yield event.plain_result(
            "📊 话题守卫统计\n"
            f"累计结束对话：{self._stats.get('total_stops', 0)} 次\n"
            f"当前会话：{len(self.sessions)} 个（进行中 {active} / 已停止 {stopped}）\n"
            f"累计对方消息：{total_turns} 条\n"
            f"结束原因分布：{reason_str}"
        )
        event.stop_event()

    @filter.command("tguard", alias={"话题守卫", "话题帮助"})
    async def cmd_help(self, event: AstrMessageEvent):
        '''话题守卫帮助。'''
        yield event.plain_result(
            "🛡️ 话题守卫（自动话题结束检测）\n"
            "/tstop 手动结束当前话题（话题停止）\n"
            "/tresume 恢复对话（话题恢复）\n"
            "/tstatus 查看会话状态（话题状态）\n"
            "/treset 重置话题与历史（话题重置）\n"
            "/tstat 查看统计（话题统计）"
        )
        event.stop_event()

    async def terminate(self):
        try:
            if self._sweeper_task:
                self._sweeper_task.cancel()
        except Exception:
            pass
        self._save_state()
        logger.info("[topic_guard] 已卸载，状态已保存")
