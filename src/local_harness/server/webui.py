"""A single-file web client for the session server.

Served at GET / by the same Starlette app that runs the SSE bus, so it's
same-origin (no CORS) and works over `lo serve` / `lo daemon`. Vanilla
JS — no build step. Full control: list/start sessions, stream live, send
follow-ups, interrupt, and approve/deny tool calls from the browser. Themed to
match the TUI's Osaka Jade palette.
"""

from __future__ import annotations

# Palette mirrors render.THEMES["osaka-jade"] so the browser matches the TUI.
_INDEX_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>local_harness</title>
<style>
  :root {
    --bg:#1a2722; --surface:#21302a; --panel:#283a32; --border:#3a9d8c;
    --jade:#52cc9e; --gold:#dcbb7a; --sakura:#e8a6c2; --rose:#e6788c;
    --cream:#cfe0d6; --grey:#7e978c; --amber:#e3a366;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--cream);
         font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; height:100vh; }
  #app { display:grid; grid-template-columns:260px 1fr; height:100vh; }
  #side { background:var(--surface); border-right:1px solid var(--border);
          display:flex; flex-direction:column; min-width:0; }
  #side h1 { font-size:15px; margin:0; padding:12px 14px; color:var(--jade);
             border-bottom:1px solid var(--border); }
  #side h1 small { color:var(--grey); font-weight:normal; }
  #newbox { padding:10px; border-bottom:1px solid var(--border); }
  #sessions { overflow:auto; flex:1; }
  .sess { padding:8px 14px; cursor:pointer; border-bottom:1px solid #122; }
  .sess:hover { background:var(--panel); }
  .sess.active { background:var(--panel); border-left:3px solid var(--jade); }
  .sess .task { color:var(--cream); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .sess .meta { color:var(--grey); font-size:12px; }
  .st-running { color:var(--gold); } .st-completed { color:var(--jade); } .st-failed { color:var(--rose); }
  #main { display:flex; flex-direction:column; min-width:0; }
  #log { overflow:auto; flex:1; padding:14px; }
  .row { margin:0 0 10px; white-space:pre-wrap; word-break:break-word; }
  .user { color:var(--jade); font-weight:bold; }
  .reason { color:var(--sakura); font-style:italic; opacity:.85; }
  .answer { color:var(--cream); }
  .tool { color:var(--gold); }
  .toolres { color:var(--grey); }
  .err { color:var(--rose); font-weight:bold; }
  .note { color:var(--amber); }
  .spawn { color:var(--jade); }
  .cursor::after { content:"▌"; color:var(--amber); }
  #perm { padding:10px 14px; background:var(--panel); border-top:1px solid var(--border); display:none; }
  #perm button { margin-left:8px; }
  #bottom { padding:10px; border-top:1px solid var(--border); display:flex; gap:8px; background:var(--surface); }
  input,button { font:inherit; }
  input { flex:1; background:var(--bg); color:var(--cream); border:1px solid var(--border);
          padding:8px; border-radius:4px; }
  button { background:var(--jade); color:#101915; border:0; padding:8px 14px; border-radius:4px;
           cursor:pointer; font-weight:bold; }
  button.ghost { background:transparent; color:var(--grey); border:1px solid var(--border); }
  button.deny { background:var(--rose); }
  #status { padding:4px 14px; color:var(--grey); font-size:12px; border-top:1px solid var(--border); }
</style>
</head>
<body>
<div id="app">
  <div id="side">
    <h1>local_harness <small id="health">·</small></h1>
    <div id="newbox"><input id="newtask" placeholder="new task… (Enter)"></div>
    <div id="sessions"></div>
  </div>
  <div id="main">
    <div id="log"><div class="row note">pick a session, or start a new task →</div></div>
    <div id="perm"></div>
    <div id="bottom">
      <input id="msg" placeholder="send a message…" disabled>
      <button id="send" disabled>send</button>
      <button id="stop" class="ghost">interrupt</button>
    </div>
    <div id="status">connecting…</div>
  </div>
</div>
<script>
let active=null, es=null, liveEl=null, reasonEl=null;
const $=s=>document.querySelector(s), log=$("#log");
const esc=s=>(s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function add(cls, html){ const d=document.createElement("div"); d.className="row "+cls; d.innerHTML=html; log.appendChild(d); log.scrollTop=log.scrollHeight; return d; }
function setStatus(t){ $("#status").textContent=t; }

async function loadSessions(){
  const r=await fetch("./sessions"); const s=await r.json();
  const box=$("#sessions"); box.innerHTML="";
  s.slice().reverse().forEach(x=>{
    const d=document.createElement("div"); d.className="sess"+(x.run_id===active?" active":"");
    d.innerHTML=`<div class="task">${esc(x.task)}</div><div class="meta">`+
      `<span class="st-${x.status}">${x.status}</span> · ${x.run_id.slice(0,8)}`+
      (x.running?" · live":"")+`</div>`;
    d.onclick=()=>select(x.run_id); box.appendChild(d);
  });
}
function finishLive(){ if(liveEl){liveEl.classList.remove("cursor");} liveEl=null; reasonEl=null; }

function select(id){
  active=id; if(es) es.close(); log.innerHTML=""; finishLive(); hidePerm();
  $("#msg").disabled=false; $("#send").disabled=false;
  es=new EventSource("./session/"+id+"/events?replay=1");
  const on=(t,f)=>es.addEventListener(t,e=>f(JSON.parse(e.data).payload||{}));
  on("run_started",p=>add("user","› "+esc(p.task)));
  on("user_message",p=>{finishLive(); add("user","› "+esc(p.content));});
  on("reasoning_delta",p=>{ if(!reasonEl) reasonEl=add("reason","✎ "); reasonEl.innerHTML+=esc(p.text); log.scrollTop=log.scrollHeight; });
  on("token_delta",p=>{ if(!liveEl){liveEl=add("answer cursor","⏺ ");} liveEl.innerHTML+=esc(p.text); log.scrollTop=log.scrollHeight; });
  on("tool_progress",p=>{ if(p.phase==="start") setStatus("running "+p.name+"…"); else setStatus("live"); });
  on("model_call",p=>{ finishLive();
    const m=((p.response||{}).choices||[{}])[0].message||{};
    const c=(m.content||"").trim(); if(c) add("answer","⏺ "+esc(c));
    (m.tool_calls||[]).forEach(tc=>add("tool","⚙ "+esc(tc.function.name)+"("+esc((tc.function.arguments||"").slice(0,80))+")"));
  });
  on("tool_call",p=>add("toolres","  ↳ "+esc(p.name)+" → "+esc((p.result||"").slice(0,200))));
  on("agent_spawned",p=>add("spawn","⑂ spawned worker "+p.child_run_id.slice(0,8)+": "+esc(p.task)));
  on("notice",p=>add("note","⚠ "+esc(p.message)));
  on("permission_request",p=>showPerm(p));
  on("context_compacted",p=>add("note","⛁ compacted "+(p.before_tokens||0)+" → "+(p.after_tokens||0)+" tokens"));
  on("run_completed",p=>{finishLive(); setStatus("done"); loadSessions();});
  on("run_failed",p=>{finishLive(); add("err","✗ "+esc(p.error)); loadSessions();});
  es.onerror=()=>setStatus("stream closed");
  setStatus("live"); loadSessions();
}

function showPerm(p){ const d=$("#perm"); d.style.display="block";
  d.innerHTML=`Allow <b>${esc(p.tool)}</b>? <code>${esc((p.arguments||"").slice(0,120))}</code>`+
    `<button onclick="respond('${p.request_id}',true)">allow</button>`+
    `<button class="deny" onclick="respond('${p.request_id}',false)">deny</button>`;
}
function hidePerm(){ $("#perm").style.display="none"; }
async function respond(rid,ok){ hidePerm();
  await fetch("./session/"+active+"/permission",{method:"POST",headers:{"content-type":"application/json"},
    body:JSON.stringify({request_id:rid,approved:ok})}); }

async function newTask(t){ const r=await fetch("./session",{method:"POST",headers:{"content-type":"application/json"},
    body:JSON.stringify({task:t})}); const j=await r.json(); await loadSessions(); select(j.run_id); }
async function send(t){ await fetch("./session/"+active+"/message",{method:"POST",
    headers:{"content-type":"application/json"},body:JSON.stringify({content:t})}); }

$("#newtask").addEventListener("keydown",e=>{ if(e.key==="Enter"&&e.target.value.trim()){ newTask(e.target.value.trim()); e.target.value=""; }});
$("#msg").addEventListener("keydown",e=>{ if(e.key==="Enter"&&e.target.value.trim()&&active){ send(e.target.value.trim()); add("user","› "+esc(e.target.value.trim())); e.target.value=""; }});
$("#send").onclick=()=>{ const v=$("#msg").value.trim(); if(v&&active){ send(v); add("user","› "+esc(v)); $("#msg").value=""; }};
$("#stop").onclick=()=>{ if(active) fetch("./session/"+active+"/interrupt",{method:"POST"}); };

async function health(){ try{ const h=await (await fetch("./health")).json();
  $("#health").textContent="· "+(h.model||"")+" · tier "+((h.capabilities||{}).tier??"?"); }catch(e){} }
health(); loadSessions(); setInterval(loadSessions,3000);
</script>
</body>
</html>
"""


def index_html() -> str:
    return _INDEX_HTML
