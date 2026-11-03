(function(){
  "use strict";
  var content=document.getElementById("content");
  if(!content)return;
  var viewer=document.createElement("section");
  viewer.id="browser-view";
  viewer.hidden=true;
  viewer.setAttribute("aria-label","Virtuális böngésző");
  viewer.innerHTML='<div id="browser-toolbar"><button id="browser-toggle" type="button" aria-expanded="false"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8M12 17v4"/></svg>Böngésző</button><span id="browser-state"></span><select id="browser-tabs" aria-label="Böngésző kiválasztása" hidden></select></div><div id="browser-stage" hidden><canvas id="browser-canvas" width="1280" height="1029" tabindex="0" aria-label="Élő felhőböngésző"></canvas><button id="browser-control" type="button">Irányítás átvétele</button></div><div id="browser-empty" hidden>Nincs aktív felhőböngésző.</div>';
  content.insertBefore(viewer,content.firstChild);
  var status=document.createElement("div");
  status.id="work-progress";
  status.hidden=true;
  status.setAttribute("role","status");
  status.setAttribute("aria-live","polite");
  status.innerHTML='<span id="work-progress-text"></span><span id="work-progress-time"></span>';
  var composer=document.getElementById("ib-wrap");
  if(composer&&composer.parentNode===content)content.insertBefore(status,composer);
  else content.appendChild(status);
  var stage=document.getElementById("browser-stage"),canvas=document.getElementById("browser-canvas"),control=document.getElementById("browser-control"),state=document.getElementById("browser-state"),toggle=document.getElementById("browser-toggle"),tabs=document.getElementById("browser-tabs"),empty=document.getElementById("browser-empty");
  var drawing=canvas.getContext("2d");
  var session=null,conversation=undefined,opened=false,activeSession=false,mode="agent",lastStatus=null,activeFrames=0,generation=0,frameSeq=0,inputQueue=Promise.resolve(),progressJob=null,controlPending=false,discovering=false,manualSelection=false;
  var mobileInput=null,typeButton=null;
  var clickState={time:0,count:0,x:0,y:0};
  var remembered=null;
  try{remembered=sessionStorage.getItem("chat_live_conversation");}catch(error){remembered=null;}
  function resetProgress(){
    progressJob=null;
    status.hidden=true;
  }
  function work(event){
    if(!event||typeof event!=="object")return;
    if(event.type==="progress"){
      progressJob=event.task_id||progressJob;
      status.hidden=false;
      document.getElementById("work-progress-text").textContent=typeof event.message==="string"?event.message:"";
      var total=Math.max(0,Math.floor((Number(event.elapsed_ms)||0)/1000));
      document.getElementById("work-progress-time").textContent=Math.floor(total/60)+":"+String(total%60).padStart(2,"0");
    }
    if(event.type==="done"||event.type==="terminal"||event.type==="error"){
      if(!progressJob||progressJob===event.task_id)resetProgress();
    }
    if(event.type==="agent_event"&&event.event&&event.event.type==="tool_result"&&event.event.tool==="cloud_browser_start"){
      var output=event.event.output;
      var parsed=output;
      if(typeof output==="string"){
        try{parsed=JSON.parse(output);}catch(error){parsed=null;}
      }
      if(parsed&&typeof parsed==="object"){
        var result=parsed.result&&typeof parsed.result==="object"?parsed.result:parsed;
        if(result&&typeof result.viewer_id==="string"&&result.viewer_id){
          selectSession(result.viewer_id);
          activeSession=true;
          opened=true;
          manualSelection=false;
          show();
        }
      }
    }
    if(event.type==="agent_event"&&event.event&&event.event.type==="tool_result"&&event.event.tool==="cloud_browser_stop"){
      activeSession=false;
      opened=false;
      selectSession(null);
      show();
    }
  }
  window.chatWorkEvent=work;
  function show(){
    viewer.hidden=!activeSession;
    toggle.setAttribute("aria-expanded",String(opened));
    stage.hidden=!opened||!session||!activeSession;
    empty.hidden=!opened||!!session||!activeSession;
  }
  function applyMode(){
    var human=mode==="human";
    var active=lastStatus==="active";
    stage.classList.toggle("human",human);
    control.disabled=controlPending||!active;
    control.textContent=controlPending?"Átadás folyamatban…":human?"Visszaadom az AI-nak":"Irányítás átvétele";
    state.textContent=!active?(lastStatus===null?"":"Böngésző leállítva"):human?"Te irányítasz":"AI dolgozik";
    if(typeButton)typeButton.setAttribute("aria-pressed",human?"true":"false");
    if(!human&&mobileInput){
      mobileInput.hidden=true;
      mobileInput.value="";
    }
  }
  function selectSession(id){
    if(session===id)return;
    session=id;
    generation++;
    mode="agent";
    lastStatus=null;
    controlPending=false;
    progressJob=null;
    delete canvas.dataset.frame;
    if(drawing)drawing.clearRect(0,0,canvas.width,canvas.height);
    if(mobileInput){
      mobileInput.hidden=true;
      mobileInput.value="";
    }
    if(typeButton)typeButton.setAttribute("aria-pressed","false");
    if(!id){
      opened=false;
      activeSession=false;
    }
    state.textContent=id?"Kapcsolódás…":"";
    control.disabled=true;
    control.textContent="Irányítás átvétele";
    stage.classList.remove("human");
    show();
  }
  toggle.addEventListener("click",function(){opened=!opened;show();});
  tabs.addEventListener("change",function(){manualSelection=true;selectSession(tabs.value);activeSession=true;opened=true;show();});
  function readToken(){
    try{
      if(typeof window.chatAccessToken==="string"&&window.chatAccessToken)return window.chatAccessToken;
      var token=window.localStorage.getItem("chat_access_token");
      if(!token)token=window.sessionStorage.getItem("chat_access_token");
      return typeof token==="string"&&token?token:null;
    }catch(error){return null;}
  }
  function errorMessage(payload){
    if(payload&&typeof payload.error==="string")return payload.error;
    if(payload&&payload.error&&typeof payload.error.message==="string")return payload.error.message;
    return "A böngésző nem érhető el.";
  }
  async function request(url,options){
    var settings=Object.assign({credentials:"same-origin"},options||{});
    var headers=Object.assign({},settings.headers||{});
    var token=readToken();
    if(token&&!headers.Authorization)headers.Authorization="Bearer "+token;
    settings.headers=headers;
    var timer=null;
    if(!settings.signal){
      if(typeof AbortSignal!=="undefined"&&typeof AbortSignal.timeout==="function"){
        settings.signal=AbortSignal.timeout(18000);
      }else if(typeof AbortController!=="undefined"){
        var controller=new AbortController();
        settings.signal=controller.signal;
        timer=window.setTimeout(function(){controller.abort();},18000);
      }
    }
    try{
      var response=await fetch(url,settings);
      var text=await response.text();
      var payload=null;
      if(text){
        try{payload=JSON.parse(text);}catch(parseError){payload=null;}
      }
      if(!response.ok)throw new Error(errorMessage(payload));
      if(payload===null)throw new Error("A böngésző nem érhető el.");
      return payload;
    }finally{
      if(timer!==null)window.clearTimeout(timer);
    }
  }
  function findSession(sessions,id){
    for(var index=0;index<sessions.length;index++)if(sessions[index].id===id)return sessions[index];
    return null;
  }
  async function discover(){
    if(discovering)return;
    discovering=true;
    var id=null;
    try{
      var app=window.chatApp;
      if(!app||typeof app.getConversationId!=="function")return;
      id=app.getConversationId()||null;
      if(!id&&remembered&&typeof app.getHistory==="function"&&typeof app.loadConversation==="function"){
        var history=app.getHistory();
        if(Object.prototype.toString.call(history)==="[object Array]"){
          var entry=null;
          for(var index=0;index<history.length;index++){
            if(history[index]&&history[index].conversation_id===remembered){entry=history[index];break;}
          }
          if(entry){
            app.loadConversation(entry.id);
            remembered=null;
            id=app.getConversationId()||null;
          }
        }
      }
      if(id!==conversation){
        conversation=id;
        selectSession(null);
        resetProgress();
        manualSelection=false;
        try{
          if(id)sessionStorage.setItem("chat_live_conversation",id);
          else sessionStorage.removeItem("chat_live_conversation");
        }catch(storageError){}
      }
      if(!id){
        show();
        return;
      }
      var result=await request("/api/browser?conversation="+encodeURIComponent(id));
      if(conversation!==id)return;
      var raw=result.sessions;
      var sessions=[];
      if(Object.prototype.toString.call(raw)==="[object Array]"){
        for(var item=0;item<raw.length;item++)if(raw[item]&&typeof raw[item].id==="string")sessions.push(raw[item]);
      }
      tabs.replaceChildren();
       sessions.filter(function(entry){return entry.status==="active";}).forEach(function(entry,index){
        var option=document.createElement("option");
         option.value=entry.id;
         option.textContent="Böngésző "+(index+1);
        tabs.appendChild(option);
      });
       var activeSessions=sessions.filter(function(entry){return entry.status==="active";});
       tabs.hidden=activeSessions.length<2;
      var active=null;
       if(activeSessions.length)active=activeSessions[0];
      var current=findSession(sessions,session);
       if(!active){
         activeSession=false;
         opened=false;
         if(session)selectSession(null);
       }else if(!current||current.status!=="active"){
         selectSession(active.id);
         activeSession=true;
         opened=true;
         manualSelection=false;
       }else{
         activeSession=true;
      }
      if(!session)opened=false;
      if(session&&tabs.value!==session)tabs.value=session;
      show();
    }catch(error){
      if(conversation===id&&id)state.textContent=error.message;
    }finally{
      discovering=false;
    }
  }
  async function frame(force){
    if(!session||!opened||document.hidden)return;
    if(activeFrames>0&&!force)return;
    activeFrames++;
    var id=session,version=generation;
    try{
      var result=await request("/api/browser?id="+encodeURIComponent(id)+"&frame=1");
      if(version!==generation)return;
      lastStatus=typeof result.status==="string"?result.status:"unknown";
      if(!controlPending)mode=typeof result.control_mode==="string"?result.control_mode:"agent";
      applyMode();
      if(result.data){
        frameSeq++;
        var sequence=frameSeq;
        var image=new Image();
        image.onload=function(){
          if(version!==generation||sequence!==frameSeq)return;
          if(!drawing)return;
          drawing.drawImage(image,0,0,1280,1029);
          canvas.dataset.frame="ready";
        };
        image.onerror=function(){
          if(version!==generation||sequence!==frameSeq)return;
          canvas.dataset.frame="error";
        };
        image.src="data:image/jpeg;base64,"+result.data;
      }
    }catch(error){
      if(version===generation)state.textContent=error.message;
    }finally{
      activeFrames=Math.max(0,activeFrames-1);
      if(version!==generation&&session)frame();
    }
  }
  control.addEventListener("click",async function(){
    if(!session||controlPending)return;
    var id=session,version=generation;
    controlPending=true;
    applyMode();
    try{
      await inputQueue;
      if(version!==generation)return;
      var target=mode==="human"?"agent":"human";
      await request("/api/browser",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:id,mode:target})});
      if(version!==generation)return;
      await frame(true);
      if(version!==generation)return;
      if(mode==="human"&&typeof canvas.focus==="function")canvas.focus();
    }catch(error){
      if(version===generation)state.textContent=error.message;
    }finally{
      if(version===generation)controlPending=false;
      applyMode();
    }
  });
  function clamp(value,minimum,maximum){
    if(value===Infinity)return maximum;
    if(!isFinite(value))return minimum;
    return Math.max(minimum,Math.min(maximum,value));
  }
  function point(event){
    var rect=canvas.getBoundingClientRect();
    var width=rect.width>0?rect.width:1280;
    var height=rect.height>0?rect.height:1029;
    return {
      x:clamp((event.clientX-rect.left)*1280/width,0,1280),
      y:clamp((event.clientY-rect.top)*1029/height,0,1029)
    };
  }
  function clickCountFor(coords,type){
    if(type!=="mousePressed"&&type!=="mouseReleased")return 0;
    var now=Date.now();
    var near=Math.abs(coords.x-clickState.x)<4&&Math.abs(coords.y-clickState.y)<4;
    if(type==="mousePressed"){
      if(now-clickState.time<500&&near)clickState.count=Math.min(3,clickState.count+1);
      else clickState.count=1;
      clickState.time=now;
      clickState.x=coords.x;
      clickState.y=coords.y;
    }
    return clickState.count||1;
  }
  function mouse(event,type){
    if(mode!=="human"||controlPending)return;
    if(!Number.isFinite(event.clientX)||!Number.isFinite(event.clientY))return;
    event.preventDefault();
    var coords=point(event);
    input({
      kind:"mouse",
      type:type,
      button:["left","middle","right"][event.button]||"none",
      buttons:event.buttons,
      clickCount:clickCountFor(coords,type),
      x:coords.x,
      y:coords.y
    });
  }
  function wheelPixels(event,delta,pageSize){
    if(event.deltaMode===1)return delta*16;
    if(event.deltaMode===2)return delta*pageSize;
    return delta;
  }
  function keyPayload(type,event){
    return {
      kind:"key",
      type:type,
      key:event.key,
      code:event.code,
      keyCode:event.keyCode,
      modifiers:(event.altKey?1:0)|(event.ctrlKey?2:0)|(event.metaKey?4:0)|(event.shiftKey?8:0)
    };
  }
  function isPrintable(event){
    if(event.ctrlKey||event.metaKey||event.altKey)return false;
    var characters=Array.from(event.key||"");
    return characters.length===1;
  }
  function isPasteShortcut(event){
    return (event.ctrlKey||event.metaKey)&&String(event.key||"").toLowerCase()==="v";
  }
  canvas.addEventListener("pointerdown",function(event){
    if(mode!=="human")return;
    canvas.focus();
    if(typeof canvas.setPointerCapture==="function")canvas.setPointerCapture(event.pointerId);
    mouse(event,"mousePressed");
  });
  canvas.addEventListener("pointerup",function(event){mouse(event,"mouseReleased");});
  canvas.addEventListener("pointercancel",function(event){mouse(event,"mouseReleased");});
  var lastMove=0;
  canvas.addEventListener("pointermove",function(event){
    if(Date.now()-lastMove<120)return;
    lastMove=Date.now();
    mouse(event,"mouseMoved");
  });
  canvas.addEventListener("contextmenu",function(event){if(mode==="human")event.preventDefault();});
  canvas.addEventListener("wheel",function(event){
    if(mode!=="human"||controlPending)return;
    if(!Number.isFinite(event.clientX)||!Number.isFinite(event.clientY))return;
    event.preventDefault();
    var coords=point(event);
    var pageSize=canvas.getBoundingClientRect().height>0?canvas.getBoundingClientRect().height:1029;
    input({
      kind:"wheel",
      type:"mouseWheel",
      deltaX:clamp(wheelPixels(event,event.deltaX,1280),-10000,10000),
      deltaY:clamp(wheelPixels(event,event.deltaY,pageSize),-10000,10000),
      x:coords.x,
      y:coords.y
    });
  },{passive:false});
  canvas.addEventListener("keydown",function(event){
    if(mode!=="human"||controlPending)return;
    if(isPasteShortcut(event))return;
    if(event.key==="Tab")return;
    if(isPrintable(event)){
      event.preventDefault();
      input({kind:"text",text:event.key});
      return;
    }
    if(event.key!=="Escape")event.preventDefault();
    input(keyPayload("keyDown",event));
  });
  canvas.addEventListener("keyup",function(event){
    if(mode!=="human"||controlPending)return;
    if(isPasteShortcut(event))return;
    if(isPrintable(event))return;
    if(event.key==="Tab")return;
    input(keyPayload("keyUp",event));
  });
  canvas.addEventListener("paste",function(event){
    if(mode!=="human")return;
    if(!event.clipboardData)return;
    event.preventDefault();
    var text=event.clipboardData.getData("text/plain");
    if(text)input({kind:"text",text:text});
  });
  mobileInput=document.createElement("input");
  mobileInput.type="text";
  mobileInput.autocomplete="off";
  mobileInput.setAttribute("aria-label","Szöveg a felhőböngészőbe");
  mobileInput.hidden=true;
  var toolbar=document.getElementById("browser-toolbar");
  if(toolbar)toolbar.appendChild(mobileInput);
  typeButton=document.createElement("button");
  typeButton.type="button";
  typeButton.textContent="⌨";
  typeButton.title="Szöveg beírása";
  typeButton.setAttribute("aria-label","Szöveg beírása");
  typeButton.setAttribute("aria-pressed","false");
  if(toolbar)toolbar.appendChild(typeButton);
  typeButton.addEventListener("click",function(){
    if(mode!=="human")return;
    mobileInput.hidden=!mobileInput.hidden;
    if(!mobileInput.hidden)mobileInput.focus();
  });
  mobileInput.addEventListener("keydown",function(event){
    if(event.key!=="Enter")return;
    event.preventDefault();
    if(mobileInput.value)input({kind:"text",text:mobileInput.value});
    mobileInput.value="";
    mobileInput.hidden=true;
    canvas.focus();
  });
  function input(data){
    if(mode!=="human"||controlPending||!session)return inputQueue;
    var id=session,version=generation;
    inputQueue=inputQueue.then(function(){
      if(mode!=="human"||controlPending||session!==id||version!==generation)return null;
      return request("/api/browser",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:id,input:data})});
    }).catch(function(error){
      if(version===generation)state.textContent=error.message;
      return null;
    });
    return inputQueue;
  }
  window.setInterval(discover,3000);
  window.setInterval(frame,900);
  document.addEventListener("visibilitychange",function(){
    if(!document.hidden){discover();frame();}
  });
  discover();
})();
