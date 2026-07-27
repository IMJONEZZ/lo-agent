"""A single-file web client for the session server.

Served at GET / by the same Starlette app that runs the SSE bus, so it's
same-origin (no CORS) and works over `lo serve` / `lo daemon`. Vanilla
JS — no build step. Full control: list/start sessions, stream live, send
follow-ups, interrupt, and approve/deny tool calls from the browser. Themed to
match the TUI's Osaka Jade palette.

Phone-first: the layout is a single column with the session list as an
off-canvas drawer, and it widens into the two-pane desktop view at 760px. The
mobile details that matter are load-bearing, not decoration — 16px inputs (iOS
zooms the page on anything smaller), `dvh` + visualViewport sizing (so the
composer rides above the on-screen keyboard), `env(safe-area-inset-*)` padding
(notch and home indicator), 44px tap targets, and an explicit SSE reconnect
(backgrounding a mobile tab kills the stream, and the browser's own retry would
replay the log on top of itself). `/manifest.webmanifest` + `/icon.svg` make
"add to home screen" open it standalone.
"""

from __future__ import annotations

import json

# Palette mirrors render.THEMES["osaka-jade"] so the browser matches the TUI.
_BG = "#1a2722"
_JADE = "#52cc9e"

_INDEX_HTML = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="__BG__">
<meta name="color-scheme" content="dark">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="lo">
<link rel="manifest" href="./manifest.webmanifest">
<link rel="icon" href="./icon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="./icon.svg">
<title>local_harness</title>
<style>
  :root {
    --bg:#1a2722; --surface:#21302a; --panel:#283a32; --border:#3a9d8c;
    --jade:#52cc9e; --gold:#dcbb7a; --sakura:#e8a6c2; --rose:#e6788c;
    --cream:#cfe0d6; --grey:#7e978c; --amber:#e3a366;
    --tap:44px;                    /* minimum comfortable touch target */
    --app-h:100vh;                 /* JS narrows this to the visual viewport */
    --safe-b:env(safe-area-inset-bottom,0px);
    --safe-t:env(safe-area-inset-top,0px);
    --safe-l:env(safe-area-inset-left,0px);
  }
  @supports (height:100dvh) { :root { --app-h:100dvh; } }
  * { box-sizing:border-box; }
  html, body { height:100%; }
  body { margin:0; background:var(--bg); color:var(--cream);
         font:15px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
         -webkit-text-size-adjust:100%; overscroll-behavior-y:none; }
  #app { display:flex; flex-direction:column; height:var(--app-h); overflow:hidden; }

  /* ---- top bar: the only chrome that's always on screen ---- */
  #top { flex:0 0 auto; display:flex; align-items:center; gap:6px;
         padding:6px 8px 6px calc(8px + var(--safe-l)); padding-top:calc(6px + var(--safe-t));
         background:var(--surface); border-bottom:1px solid var(--border); }
  #title { flex:1; min-width:0; }
  /* one line, always: a long task title must not grow the bar to two rows */
  #title b { display:block; color:var(--jade);
             white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  #title .who { display:block; color:var(--grey); font-size:12px;
                white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .icon { background:transparent; color:var(--cream); border:1px solid var(--border);
          border-radius:8px; min-width:var(--tap); min-height:var(--tap);
          font-size:17px; line-height:1; cursor:pointer; touch-action:manipulation; }
  .icon:active { background:var(--panel); }

  #body { flex:1; display:flex; min-height:0; }

  /* ---- sessions: off-canvas drawer on phones, a static column at >=760px ---- */
  #side { position:fixed; z-index:30; top:0; bottom:0; left:0; width:min(84vw,300px);
          transform:translateX(-101%); transition:transform .18s ease;
          background:var(--surface); border-right:1px solid var(--border);
          display:flex; flex-direction:column; min-width:0;
          padding-top:var(--safe-t); padding-bottom:var(--safe-b);
          padding-left:var(--safe-l); }
  #side.open { transform:none; }
  #side h1 { font-size:15px; margin:0; padding:12px 14px; color:var(--jade);
             border-bottom:1px solid var(--border); }
  #side h1 small { color:var(--grey); font-weight:normal; }
  #scrim { position:fixed; inset:0; z-index:20; background:#0009; display:none; }
  #scrim.open { display:block; }
  #newbox { padding:10px; border-bottom:1px solid var(--border); }
  #sessions { overflow-y:auto; overscroll-behavior:contain;
              -webkit-overflow-scrolling:touch; flex:1; }
  .sess { padding:11px 14px; cursor:pointer; border-bottom:1px solid #122;
          min-height:var(--tap); touch-action:manipulation; }
  .sess:active, .sess:hover { background:var(--panel); }
  .sess.active { background:var(--panel); border-left:3px solid var(--jade); }
  .sess .task { color:var(--cream); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .sess .meta { color:var(--grey); font-size:12px; }
  .st-running { color:var(--gold); } .st-completed { color:var(--jade); } .st-failed { color:var(--rose); }

  #main { flex:1; display:flex; flex-direction:column; min-width:0; position:relative; }
  #log { overflow-y:auto; overscroll-behavior:contain; -webkit-overflow-scrolling:touch;
         flex:1; padding:12px calc(12px + var(--safe-l)) 12px 12px; }
  .row { margin:0 0 10px; white-space:pre-wrap; overflow-wrap:anywhere; }
  .user { color:var(--jade); font-weight:bold; }
  .reason { color:var(--sakura); font-style:italic; opacity:.85; }
  .answer { color:var(--cream); }
  .tool { color:var(--gold); }
  .toolres { color:var(--grey); }
  .err { color:var(--rose); font-weight:bold; }
  .note { color:var(--amber); }
  .spawn { color:var(--jade); }
  .cursor::after { content:"▌"; color:var(--amber); }

  /* ---- rendered markdown: the reason a long answer is readable on a phone ---- */
  .row.md { position:relative; padding-left:1.5em; }
  .row.md::before { content:"⏺"; position:absolute; left:0; }
  .md p { margin:0 0 8px; }
  .md p:last-child, .md ul:last-child, .md .cb:last-child { margin-bottom:0; }
  .md ul { margin:4px 0 8px; padding-left:20px; }
  .md li { margin:2px 0; }
  .md .h { color:var(--jade); font-weight:bold; margin:10px 0 4px; }
  .md blockquote { margin:6px 0; padding-left:10px; color:var(--grey);
                   border-left:2px solid var(--border); }
  .md hr { border:0; border-top:1px solid var(--border); margin:10px 0; }
  .md a { color:var(--jade); }
  /* inline code must not break mid-identifier the way prose does */
  .md code { background:#16211d; border-radius:4px; padding:1px 4px;
             color:var(--gold); overflow-wrap:normal; word-break:keep-all; }
  /* a code block scrolls sideways in its own box — wrapping it at 390px turns
     every line into an unreadable ribbon, and the page must never scroll */
  .md .cb { margin:8px 0; border:1px solid #2c4139; border-radius:8px;
            overflow:hidden; background:#16211d; }
  .md .cbh { display:flex; align-items:center; justify-content:space-between;
             gap:8px; padding:3px 5px 3px 10px; background:var(--panel);
             border-bottom:1px solid #2c4139; }
  .md .cbh .lang { color:var(--grey); font-size:12px; }
  pre.code { margin:0; padding:10px 12px; white-space:pre;
             overflow-x:auto; overscroll-behavior-x:contain;
             -webkit-overflow-scrolling:touch; }
  pre.code code { background:none; padding:0; font-size:13px; color:var(--cream);
                  white-space:pre; }
  /* not var(--tap): a 44px bar on top of every block eats a phone screen */
  .copy { min-height:32px; padding:4px 12px; font-size:12px; font-weight:normal;
          border-radius:6px; background:transparent; color:var(--jade);
          border:1px solid var(--border); }
  /* tool output: a tappable "show all" beats silently swallowing 40kB */
  .more { min-height:26px; padding:2px 8px; margin-left:6px; font-size:12px;
          font-weight:normal; border-radius:6px; background:transparent;
          color:var(--jade); border:1px solid var(--border); }
  .toolres.open .full { display:block; max-height:40vh; overflow:auto;
                        overscroll-behavior:contain; margin-top:4px; }

  /* jump-back-to-live pill: reading scrollback shouldn't get yanked away */
  #jump { position:absolute; left:50%; transform:translateX(-50%); bottom:8px;
          display:none; z-index:10; background:var(--panel); color:var(--jade);
          border:1px solid var(--border); border-radius:999px; padding:8px 16px;
          min-height:36px; cursor:pointer; touch-action:manipulation; }
  #jump.show { display:block; }

  #perm { padding:12px calc(12px + var(--safe-l)) 12px 12px; background:var(--panel);
          border-top:1px solid var(--border); display:none; }
  #perm.show { display:block; }
  #perm code { display:block; color:var(--grey); margin:6px 0 10px;
               overflow-wrap:anywhere; }
  #permbtns { display:flex; gap:8px; }
  #permbtns button { flex:1; }

  /* the composer is the bottom-most element, so it carries the home-indicator inset */
  #bottom { flex:0 0 auto; display:flex; gap:8px; align-items:flex-end;
            padding:8px calc(8px + var(--safe-l)) calc(8px + var(--safe-b)) 8px;
            background:var(--surface); }
  input, button, textarea { font-family:inherit; }
  /* 16px is not a style choice: iOS Safari zooms the page on smaller inputs */
  input, textarea { font-size:16px; background:var(--bg); color:var(--cream);
                    border:1px solid var(--border); padding:10px; border-radius:8px;
                    width:100%; }
  textarea { flex:1; resize:none; max-height:35vh; line-height:1.4;
             min-height:var(--tap); overflow-y:auto; }
  button { background:var(--jade); color:#101915; border:0; padding:10px 16px;
           border-radius:8px; cursor:pointer; font-weight:bold; font-size:15px;
           min-height:var(--tap); touch-action:manipulation; }
  button:disabled { opacity:.45; }
  button.ghost { background:transparent; color:var(--grey); border:1px solid var(--border); }
  button.deny { background:var(--rose); }
  #send { min-width:var(--tap); }
  #status { flex:0 0 auto; padding:3px 12px; color:var(--grey);
            font-size:12px; border-top:1px solid var(--border); background:var(--surface); }

  @media (min-width:760px) {
    #side { position:static; transform:none; transition:none; width:270px; }
    #scrim { display:none !important; }
    #menu { display:none; }
    #title .who { display:inline; }
  }
  @media (prefers-reduced-motion:reduce) { #side { transition:none; } }
</style>
</head>
<body>
<div id="app">
  <header id="top">
    <button id="menu" class="icon" aria-label="sessions">&#9776;</button>
    <div id="title"><b>local_harness</b><span class="who" id="health">&middot;</span></div>
    <button id="stop" class="icon" aria-label="interrupt" title="interrupt">&#9632;</button>
  </header>
  <div id="body">
    <div id="scrim"></div>
    <aside id="side">
      <h1>sessions <small id="count"></small></h1>
      <div id="newbox"><input id="newtask" placeholder="new task&hellip;" enterkeyhint="go"
                              autocapitalize="sentences" autocomplete="off"></div>
      <div id="sessions"></div>
    </aside>
    <main id="main">
      <div id="log"><div class="row note">pick a session from the menu, or start a new task</div></div>
      <button id="jump">&darr; latest</button>
      <div id="perm"></div>
      <div id="status">connecting&hellip;</div>
      <div id="bottom">
        <textarea id="msg" rows="1" placeholder="send a message&hellip;" disabled
                  enterkeyhint="send" autocapitalize="sentences"></textarea>
        <button id="send" disabled aria-label="send">&uarr;</button>
      </div>
    </main>
  </div>
</div>
<script>
/* --- markdown for the transcript -----------------------------------------
   Deliberately pure: no DOM, no app globals, so the test suite can run it
   under node. Everything model- or tool-authored reaches the output through
   esc(); the one attribute ever emitted is an href matched against an http(s)
   pattern that cannot contain a quote or an angle bracket, so there is no
   path from generated text into markup. Kept small on purpose — headings,
   lists, quotes, rules, fences, and inline code/bold/italic/links. */
function esc(s){ return (s==null?"":String(s)).replace(/[&<>]/g,
  c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

function mdInline(t){
  // `code` first so formatting inside backticks stays literal. No _italic_:
  // snake_case identifiers are far more common here than emphasis.
  const re=/`([^`]+)`|\\*\\*([\\s\\S]+?)\\*\\*|\\*([^*\\n]+)\\*|\\[([^\\]\\n]+)\\]\\((https?:\\/\\/[^\\s)<>"']+)\\)/g;
  let out="", last=0, m;
  while((m=re.exec(t))!==null){
    out+=esc(t.slice(last,m.index));
    if(m[1]!==undefined) out+="<code>"+esc(m[1])+"</code>";
    else if(m[2]!==undefined) out+="<b>"+mdInline(m[2])+"</b>";
    else if(m[3]!==undefined) out+="<i>"+mdInline(m[3])+"</i>";
    else out+='<a href="'+m[5]+'" target="_blank" rel="noreferrer noopener">'
              +esc(m[4])+"</a>";
    last=re.lastIndex;
  }
  return out+esc(t.slice(last));
}

function mdCode(body, lang){
  const tag=/^[\\w+.#-]{1,20}$/.test(lang||"") ? lang : "";
  return '<div class="cb"><div class="cbh"><span class="lang">'+esc(tag)
       +'</span><button class="copy" type="button">copy</button></div>'
       +'<pre class="code"><code>'+esc(body)+"</code></pre></div>";
}

function md(src){
  const lines=String(src==null?"":src).replace(/\\r/g,"").split("\\n");
  const out=[]; let para=[], i=0;
  const flush=()=>{ if(para.length){ out.push("<p>"+mdInline(para.join("\\n"))+"</p>");
                                     para=[]; } };
  const bullet=/^\\s*([-*+]|\\d+[.)])\\s+/;
  while(i<lines.length){
    const l=lines[i], fence=l.match(/^\\s*```\\s*([^\\s`]*)\\s*$/);
    if(fence){ flush(); const buf=[]; i++;
      while(i<lines.length && !/^\\s*```/.test(lines[i])) buf.push(lines[i++]);
      i++;                       // closing fence, or EOF mid-stream
      out.push(mdCode(buf.join("\\n"), fence[1])); continue; }
    const h=l.match(/^(#{1,4})\\s+(.+)$/);
    if(h){ flush(); out.push('<div class="h">'+mdInline(h[2])+"</div>"); i++; continue; }
    if(bullet.test(l)){ flush(); const items=[];
      while(i<lines.length && bullet.test(lines[i]))
        items.push("<li>"+mdInline(lines[i++].replace(bullet,""))+"</li>");
      out.push("<ul>"+items.join("")+"</ul>"); continue; }
    if(/^\\s*>\\s?/.test(l)){ flush(); const q=[];
      while(i<lines.length && /^\\s*>\\s?/.test(lines[i]))
        q.push(lines[i++].replace(/^\\s*>\\s?/,""));
      out.push("<blockquote>"+mdInline(q.join("\\n"))+"</blockquote>"); continue; }
    if(/^\\s*(-{3,}|\\*{3,}|_{3,})\\s*$/.test(l)){ flush(); out.push("<hr>"); i++; continue; }
    if(!l.trim()){ flush(); i++; continue; }
    para.push(l); i++;
  }
  flush(); return out.join("");
}
</script>
<script>
let active=null, es=null, liveEl=null, liveText=null, reasonEl=null, reasonText=null;
let retry=null, backoff=1000, sessionsById={};
const $=s=>document.querySelector(s), log=$("#log"), side=$("#side"), scrim=$("#scrim");
const near=()=>log.scrollHeight-log.scrollTop-log.clientHeight<80;
function toEnd(){ log.scrollTop=log.scrollHeight; $("#jump").classList.remove("show"); }
function settle(was){ if(was) toEnd(); else $("#jump").classList.add("show"); }
function add(cls, html){ const was=near(); const d=document.createElement("div");
  d.className="row "+cls; d.innerHTML=html; log.appendChild(d); settle(was); return d; }
// streaming deltas append to a text node — no re-parse of a growing innerHTML
function stream(el, node, text){ const was=near(); node.appendData(text); settle(was); }
function textRow(cls, prefix){ const d=add(cls, esc(prefix));
  const n=document.createTextNode(""); d.appendChild(n); return [d,n]; }
function setStatus(t){ $("#status").textContent=t; }

/* tool results are unbounded — show a readable head, keep the rest a tap away */
const CUT=200;
function toolRow(name, result){
  const was=near(), d=add("toolres","  \\u21b3 "+esc(name)+" \\u2192 ");
  const body=document.createElement("span"); body.className="full";
  const head=result.slice(0,CUT);
  body.textContent=result.length>CUT?head+"\\u2026":head;
  d.appendChild(body);
  if(result.length>CUT){
    const b=document.createElement("button"); b.className="more"; b.type="button";
    const shut="show all ("+result.length+")";
    b.textContent=shut;
    b.onclick=()=>{ const open=d.classList.toggle("open");
      body.textContent=open?result:head+"\\u2026"; b.textContent=open?"less":shut; };
    d.appendChild(b);
  }
  settle(was);
}

/* Copy on a code block. navigator.clipboard is gated to secure contexts, and
   the phone reaches this over plain http on a LAN address — so the execCommand
   fallback is the path that actually runs, not a legacy nicety. */
function copyText(text, btn){
  const label=btn.textContent, flash=()=>{ btn.textContent="copied";
    setTimeout(()=>{ btn.textContent=label; },1200); };
  if(window.isSecureContext && navigator.clipboard)
    return navigator.clipboard.writeText(text).then(flash,()=>legacyCopy(text,flash));
  legacyCopy(text,flash);
}
function legacyCopy(text, done){
  const ta=document.createElement("textarea"); ta.value=text;
  ta.style.cssText="position:fixed;top:0;left:0;opacity:0";
  document.body.appendChild(ta); ta.focus(); ta.select();
  try{ document.execCommand("copy"); done(); }catch(e){}
  ta.remove();
}
log.addEventListener("click",e=>{ const b=e.target.closest&&e.target.closest(".copy");
  if(!b) return; const code=b.closest(".cb").querySelector("code");
  if(code) copyText(code.textContent, b); });

/* ---- drawer ---- */
function openDrawer(quiet){ side.classList.add("open"); scrim.classList.add("open");
  if(!quiet){ try{ history.pushState({drawer:1},""); }catch(e){} } }
function closeDrawer(fromPop){ side.classList.remove("open"); scrim.classList.remove("open");
  if(!fromPop && history.state && history.state.drawer) history.back(); }
$("#menu").onclick=()=>side.classList.contains("open")?closeDrawer():openDrawer();
scrim.onclick=()=>closeDrawer();
addEventListener("popstate",()=>{ if(side.classList.contains("open")) closeDrawer(true); });

async function loadSessions(){
  let s; try{ s=await (await fetch("./sessions")).json(); }catch(e){ return; }
  sessionsById={}; s.forEach(x=>sessionsById[x.run_id]=x);
  $("#count").textContent=s.length?"("+s.length+")":"";
  const box=$("#sessions"); box.innerHTML="";
  s.slice().reverse().forEach(x=>{
    const d=document.createElement("div"); d.className="sess"+(x.run_id===active?" active":"");
    d.innerHTML=`<div class="task">${esc(x.task)}</div><div class="meta">`+
      `<span class="st-${x.status}">${x.status}</span> &middot; ${x.run_id.slice(0,8)}`+
      (x.running?" &middot; live":"")+`</div>`;
    d.onclick=()=>{ closeDrawer(); select(x.run_id); }; box.appendChild(d);
  });
  nameTurn();
}
function nameTurn(){ const s=sessionsById[active];
  $("#title").firstChild.textContent=s?(s.task.length>40?s.task.slice(0,40)+"\\u2026":s.task)
                                      :"local_harness";
  document.title=s?("lo \\u00b7 "+s.task.slice(0,40)):"local_harness"; }
function finishLive(){ if(liveEl) liveEl.classList.remove("cursor");
  liveEl=liveText=reasonEl=reasonText=null; }

function select(id, keep){
  active=id; try{ localStorage.setItem("lo.active",id); }catch(e){}
  if(es){ es.close(); es=null; } if(retry){ clearTimeout(retry); retry=null; }
  if(!keep) backoff=1000;
  log.innerHTML=""; finishLive(); hidePerm(); nameTurn();
  $("#msg").disabled=false; $("#send").disabled=false;
  es=new EventSource("./session/"+id+"/events?replay=1");
  const on=(t,f)=>es.addEventListener(t,e=>f(JSON.parse(e.data).payload||{}));
  on("run_started",p=>add("user","&rsaquo; "+esc(p.task)));
  on("user_message",p=>{finishLive(); add("user","&rsaquo; "+esc(p.content));});
  on("reasoning_delta",p=>{ if(!reasonEl){ [reasonEl,reasonText]=textRow("reason","\\u270e "); }
    stream(reasonEl,reasonText,p.text); });
  on("token_delta",p=>{ if(!liveEl){ [liveEl,liveText]=textRow("answer cursor","\\u23fa ");
      liveEl.classList.add("cursor"); } stream(liveEl,liveText,p.text); });
  on("tool_progress",p=>{ setStatus(p.phase==="start"?("running "+p.name+"\\u2026"):"live"); });
  on("model_call",p=>{
    const m=((p.response||{}).choices||[{}])[0].message||{};
    const c=(m.content||"").trim();
    // The same answer arrives twice — once as live token deltas, then as the
    // persisted call. Upgrade the streamed row in place (plain text while it
    // streams, markdown once it's whole) instead of printing it again.
    if(c){ const was=near(); const row=liveEl||add("answer","");
           row.className="row answer md"; row.innerHTML=md(c); settle(was); }
    finishLive();
    (m.tool_calls||[]).forEach(tc=>add("tool","\\u2699 "+esc(tc.function.name)+"("+esc((tc.function.arguments||"").slice(0,120))+")"));
  });
  on("tool_call",p=>toolRow(p.name,p.result||""));
  on("agent_spawned",p=>add("spawn","\\u2442 spawned worker "+p.child_run_id.slice(0,8)+": "+esc(p.task)));
  on("notice",p=>add("note","\\u26a0 "+esc(p.message)));
  on("permission_request",p=>showPerm(p));
  on("context_compacted",p=>add("note","\\u26c1 compacted "+(p.before_tokens||0)+" \\u2192 "+(p.after_tokens||0)+" tokens"));
  on("run_completed",p=>{finishLive(); setStatus("done"); loadSessions();});
  on("run_failed",p=>{finishLive(); add("err","\\u2717 "+esc(p.error)); loadSessions();});
  es.onopen=()=>{ backoff=1000; setStatus("live"); };
  // Own the retry: EventSource would reconnect by itself and replay the whole
  // log on top of what's already rendered. Close, wait, then re-select (which
  // clears and replays cleanly) — this is the normal path when a phone
  // backgrounds the tab.
  es.onerror=()=>{ if(!es) return; es.close(); es=null;
    setStatus("reconnecting\\u2026");
    retry=setTimeout(()=>{ if(active===id) select(id,true); }, backoff);
    backoff=Math.min(backoff*2,15000); };
  setStatus("live"); loadSessions();
}
// a backgrounded tab gets its stream reaped; come back live on return
addEventListener("visibilitychange",()=>{ if(document.visibilityState==="visible"&&active&&!es){
  if(retry) clearTimeout(retry); retry=null; select(active,true); }});

function showPerm(p){ const d=$("#perm"); d.classList.add("show");
  d.innerHTML=`Allow <b>${esc(p.tool)}</b>?<code>${esc((p.arguments||"").slice(0,200))}</code>`+
    `<div id="permbtns"><button class="deny" id="pdeny">deny</button>`+
    `<button id="pallow">allow</button></div>`;
  $("#pallow").onclick=()=>respond(p.request_id,true);
  $("#pdeny").onclick=()=>respond(p.request_id,false);
  if(navigator.vibrate) navigator.vibrate(12);
  add("note","\\u26a0 approval needed: "+esc(p.tool));
}
function hidePerm(){ $("#perm").classList.remove("show"); }
async function respond(rid,ok){ hidePerm();
  await fetch("./session/"+active+"/permission",{method:"POST",headers:{"content-type":"application/json"},
    body:JSON.stringify({request_id:rid,approved:ok})}); }

async function newTask(t){ const r=await fetch("./session",{method:"POST",headers:{"content-type":"application/json"},
    body:JSON.stringify({task:t})}); const j=await r.json(); closeDrawer(); await loadSessions(); select(j.run_id); }
async function send(t){ await fetch("./session/"+active+"/message",{method:"POST",
    headers:{"content-type":"application/json"},body:JSON.stringify({content:t})}); }

const msg=$("#msg");
function grow(){ msg.style.height="auto"; msg.style.height=Math.min(msg.scrollHeight,window.innerHeight*.35)+"px"; }
function submit(){ const v=msg.value.trim(); if(!v||!active) return;
  send(v); add("user","&rsaquo; "+esc(v)); msg.value=""; grow(); toEnd(); }
msg.addEventListener("input",grow);
msg.addEventListener("keydown",e=>{ if(e.key==="Enter"&&!e.shiftKey){ e.preventDefault(); submit(); }});
$("#send").onclick=()=>{ submit(); msg.focus(); };
$("#newtask").addEventListener("keydown",e=>{ if(e.key==="Enter"&&e.target.value.trim()){
  newTask(e.target.value.trim()); e.target.value=""; e.target.blur(); }});
$("#stop").onclick=()=>{ if(active) fetch("./session/"+active+"/interrupt",{method:"POST"}); };
$("#jump").onclick=toEnd;
log.addEventListener("scroll",()=>{ if(near()) $("#jump").classList.remove("show"); });

// The on-screen keyboard shrinks the visual viewport without touching the
// layout viewport; without this the composer sits underneath it.
function fit(){ const vv=window.visualViewport;
  document.documentElement.style.setProperty("--app-h",(vv?vv.height:window.innerHeight)+"px");
  if(near()) toEnd(); }
if(window.visualViewport){ visualViewport.addEventListener("resize",fit);
  visualViewport.addEventListener("scroll",fit); }
addEventListener("orientationchange",()=>setTimeout(fit,120));
fit();

async function health(){ try{ const h=await (await fetch("./health")).json();
  $("#health").textContent=(h.model||"")+" \\u00b7 tier "+((h.capabilities||{}).tier??"?"); }catch(e){} }
health();
// A phone reloads this page constantly (tab eviction, app switch). Come back
// to the conversation you were in; only fall back to the picker if it's gone.
loadSessions().then(()=>{ if(active) return;
  let last=null; try{ last=localStorage.getItem("lo.active"); }catch(e){}
  if(last && sessionsById[last]) select(last);
  else { setStatus("no session open"); openDrawer(true); } });
setInterval(loadSessions,3000);
</script>
</body>
</html>
"""


def index_html() -> str:
    return _INDEX_HTML.replace("__BG__", _BG)


def manifest_json() -> str:
    """Web app manifest — 'add to home screen' opens the client standalone,
    without the browser's address bar eating a phone's vertical space."""
    return json.dumps({
        "name": "local_harness",
        "short_name": "lo",
        "description": "local agent harness — sessions, live stream, approvals",
        "start_url": "./",
        "scope": "./",
        "display": "standalone",
        "orientation": "portrait-primary",
        "background_color": _BG,
        "theme_color": _BG,
        "icons": [{"src": "./icon.svg", "sizes": "any",
                   "type": "image/svg+xml", "purpose": "any maskable"}],
    })


def icon_svg() -> str:
    """Home-screen icon: the jade prompt caret on the TUI's background. Sized
    with a maskable-safe margin so Android's circle crop doesn't clip it."""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192">'
        f'<rect width="192" height="192" rx="40" fill="{_BG}"/>'
        f'<path d="M56 62 L92 96 L56 130" fill="none" stroke="{_JADE}" '
        'stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<path d="M104 130 H140" stroke="{_JADE}" stroke-width="14" '
        'stroke-linecap="round"/></svg>'
    )
