#!/usr/bin/env python3
"""Web app for megakernel TTS: live streaming playback + voice library.

Run on the GPU server:
    python3 app.py            # listens on 127.0.0.1:8000

Access from your machine via SSH tunnel:
    ssh -L 8000:localhost:8000 -p <port> root@<server>
    -> open http://localhost:8000

Endpoints:
    GET  /                 — UI
    POST /generate         — body {text, speaker, language}; chunked stream of
                             raw PCM16 mono 24kHz; also saved to voices/*.wav
    GET  /voices           — JSON list of saved wavs
    GET  /voices/{name}    — serve a wav inline (audio/wav)
"""

import datetime
import io
import json
import os
import re
import threading
import sys
import wave

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

app = FastAPI(title="Qwen3-TTS Megakernel")
engine = None
engine_lock = threading.Lock()  # one generation at a time (single KV cache)


class GenRequest(BaseModel):
    text: str
    speaker: str = "ryan"
    language: str = "English"


def get_engine():
    global engine
    if engine is None:
        from qwen_tts_megakernel.engine import MegakernelTTS
        engine = MegakernelTTS()
        for _ in engine.stream("Warm up.", speaker="ryan", max_new_tokens=30):
            pass
    return engine


def save_wav(path: str, pcm16: bytes, sr: int):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16)


@app.post("/generate")
def generate(req: GenRequest):
    eng = get_engine()
    text = req.text.strip()
    if not text:
        raise HTTPException(400, "empty text")

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text[:40]).strip("_") or "voice"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{stamp}_{req.speaker}_{slug}.wav"
    fpath = os.path.join(VOICES_DIR, fname)

    def pcm_stream():
        all_pcm = []
        with engine_lock, torch.inference_mode():
            for audio, sr, _ in eng.stream(text, speaker=req.speaker,
                                           language=req.language):
                pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
                all_pcm.append(pcm)
                yield pcm
        save_wav(fpath, b"".join(all_pcm), eng.sample_rate)

    return StreamingResponse(
        pcm_stream(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(get_engine().sample_rate),
                 "X-Filename": fname},
    )


@app.get("/voices")
def list_voices():
    files = []
    for f in sorted(os.listdir(VOICES_DIR), reverse=True):
        if f.endswith(".wav"):
            st = os.stat(os.path.join(VOICES_DIR, f))
            files.append({"name": f, "size": st.st_size,
                          "mtime": datetime.datetime.fromtimestamp(st.st_mtime)
                          .strftime("%Y-%m-%d %H:%M:%S")})
    return files


@app.get("/voices/{name}")
def get_voice(name: str):
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad name")
    path = os.path.join(VOICES_DIR, name)
    if not os.path.isfile(path):
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="audio/wav",
                        headers={"Content-Disposition": "inline"})


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Qwen3-TTS Megakernel</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:780px;margin:2rem auto;padding:0 1rem;background:#111;color:#eee}
 h1{font-size:1.3rem} textarea{width:100%;height:5rem;background:#1c1c1c;color:#eee;border:1px solid #444;border-radius:8px;padding:.6rem;font-size:1rem}
 select,button{background:#1c1c1c;color:#eee;border:1px solid #444;border-radius:8px;padding:.5rem .9rem;font-size:1rem;margin:.4rem .4rem 0 0}
 button{cursor:pointer;background:#2563eb;border:none} button:disabled{background:#444}
 #status{margin:.6rem 0;color:#9ca3af;min-height:1.2rem}
 table{width:100%;border-collapse:collapse;margin-top:1rem}
 td,th{padding:.45rem .4rem;border-bottom:1px solid #2a2a2a;text-align:left;font-size:.9rem}
 .play{background:#16a34a;padding:.3rem .8rem} audio{width:100%;margin-top:.4rem}
</style></head><body>
<h1>🔊 Qwen3-TTS — megakernel streaming</h1>
<textarea id="text" placeholder="Type something to speak...">Hello! This audio is being generated and streamed live by a CUDA megakernel.</textarea>
<div>
 <select id="speaker">
  <option>ryan</option><option>serena</option><option>vivian</option><option>uncle_fu</option>
  <option>aiden</option><option>ono_anna</option><option>sohee</option><option>eric</option><option>dylan</option>
 </select>
 <select id="language">
  <option>English</option><option>Chinese</option><option>German</option><option>Italian</option>
  <option>Portuguese</option><option>Spanish</option><option>Japanese</option><option>Korean</option>
  <option>French</option><option>Russian</option>
 </select>
 <button id="go">Generate &amp; stream</button>
</div>
<div id="status"></div>
<h2 style="font-size:1.1rem">Voice library</h2>
<div id="player"></div>
<table><thead><tr><th>file</th><th>time</th><th>size</th><th></th></tr></thead>
<tbody id="lib"></tbody></table>
<script>
const $=id=>document.getElementById(id);
let ctx;
async function refreshLib(){
  const r=await fetch('/voices'); const files=await r.json();
  $('lib').innerHTML=files.map(f=>
    `<tr><td>${f.name}</td><td>${f.mtime}</td><td>${(f.size/1024).toFixed(0)} kB</td>
     <td><button class="play" onclick="playFile('${f.name}')">▶ play</button></td></tr>`).join('');
}
function playFile(name){
  $('player').innerHTML=`<audio controls autoplay src="/voices/${encodeURIComponent(name)}"></audio>`;
}
$('go').onclick=async()=>{
  const btn=$('go'); btn.disabled=true;
  $('status').textContent='generating…';
  ctx = ctx || new (window.AudioContext||window.webkitAudioContext)();
  await ctx.resume();
  const t0=performance.now();
  const resp=await fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text:$('text').value,speaker:$('speaker').value,language:$('language').value})});
  if(!resp.ok){$('status').textContent='error: '+await resp.text();btn.disabled=false;return;}
  const sr=parseInt(resp.headers.get('X-Sample-Rate')||'24000');
  const reader=resp.body.getReader();
  let playT=ctx.currentTime+0.1, leftover=new Uint8Array(0), total=0, first=true;
  while(true){
    const {done,value}=await reader.read();
    if(done)break;
    if(first){$('status').textContent=`first audio after ${(performance.now()-t0).toFixed(0)} ms — playing live…`;first=false;}
    let bytes=new Uint8Array(leftover.length+value.length);
    bytes.set(leftover); bytes.set(value,leftover.length);
    const usable=bytes.length-(bytes.length%2);
    leftover=bytes.slice(usable);
    const pcm=new Int16Array(bytes.buffer.slice(0,usable));
    if(!pcm.length)continue;
    const f32=Float32Array.from(pcm,v=>v/32768);
    const buf=ctx.createBuffer(1,f32.length,sr);
    buf.copyToChannel(f32,0);
    const src=ctx.createBufferSource(); src.buffer=buf; src.connect(ctx.destination);
    playT=Math.max(playT,ctx.currentTime+0.05);
    src.start(playT); playT+=buf.duration; total+=buf.duration;
  }
  $('status').textContent=`done — ${total.toFixed(1)}s audio, generated in ${((performance.now()-t0)/1000).toFixed(2)}s`;
  btn.disabled=false; refreshLib();
};
refreshLib();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


if __name__ == "__main__":
    get_engine()  # load model before accepting requests
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
