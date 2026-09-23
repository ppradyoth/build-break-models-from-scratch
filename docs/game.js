/* build & break models — game engine + shared mechanic helpers */
(function(){
const BBM = window.BBM = window.BBM || {};
const D = window.BBM_DATA || {};
BBM.D = D;

/* ---------------- progression state (localStorage, per-viewer) ------------- */
const KEY = "bbm_v1";
const LEVELS = [
  {n:1, slug:"level-1", title:"Tokenizer",   flag:"tokenization-is-not-canonical", d:"Bypass a filter that watches token IDs."},
  {n:2, slug:"level-2", title:"Embeddings",  flag:"similarity-is-not-relevance",   d:"Hijack a search result with a crafted passage."},
  {n:3, slug:"level-3", title:"Transformer", flag:"attention-has-no-provenance",   d:"Inject an instruction through untrusted data."},
  {n:4, slug:"level-4", title:"Sampling",    flag:"decoding-is-not-neutral",       d:"Jailbreak a refusal with the decoding dials."},
  {n:5, slug:"level-5", title:"Refusal",     flag:"refusal-is-one-direction",      d:"Delete a model's refusal with one direction."},
];
BBM.LEVELS = LEVELS;

function load(){ try{ return JSON.parse(localStorage.getItem(KEY)) || {done:{}}; }catch(e){ return {done:{}}; } }
function save(s){ try{ localStorage.setItem(KEY, JSON.stringify(s)); }catch(e){} }
BBM.state = load();
BBM.done  = n => !!BBM.state.done[n];
BBM.count = () => LEVELS.filter(l=>BBM.done(l.n)).length;
BBM.unlocked = n => n===1 || BBM.done(n-1);
BBM.complete = function(n){ BBM.state.done[n]=true; save(BBM.state); };
BBM.reset = function(){ BBM.state={done:{}}; save(BBM.state); location.reload(); };

/* ---------------- header + progress ---------------------------------------- */
BBM.mountHeader = function(){
  const done=BBM.count(), tot=LEVELS.length;
  const h=document.createElement("header"); h.className="bar";
  h.innerHTML='<div class="wrap">'+
    '<div class="brand"><a href="index.html"><b>build</b> &amp; <b>break</b> <span>models</span></a></div>'+
    '<div class="hprog"><span class="flags">'+done+'/'+tot+' flags</span>'+
    '<div class="track"><span style="width:'+(done/tot*100)+'%"></span></div>'+
    '<a class="small" href="https://github.com/ppradyoth/build-break-models-from-scratch" style="margin-left:8px">GitHub ↗</a></div></div>';
  document.body.prepend(h);
};

/* level footer nav: prev / build-it / next(locked until solved) */
BBM.mountLevelNav = function(n){
  const el=document.getElementById("lnav"); if(!el) return;
  const prev = n>1 ? '<a class="btn ghost" href="level-'+(n-1)+'.html">← Level '+(n-1)+'</a>' : '<a class="btn ghost" href="index.html">← Menu</a>';
  const repo='<a class="btn ghost" href="https://github.com/ppradyoth/build-break-models-from-scratch/tree/main/levels/level-0'+n+'-'+({1:"tokenizer",2:"embeddings",3:"transformer",4:"sampling",5:"refusal-direction"})[n]+'">Build it for real ↗</a>';
  let next;
  if(n<5) next='<a class="btn primary" id="nextbtn" href="level-'+(n+1)+'.html">Level '+(n+1)+' →</a>';
  else next='<a class="btn primary" id="nextbtn" href="victory.html">Finish →</a>';
  el.innerHTML=prev+repo+next;
  const nb=document.getElementById("nextbtn");
  if(!BBM.done(n)){ nb.classList.add("disabled"); nb.style.pointerEvents="none"; nb.style.opacity=".4"; nb.dataset.locked="1"; }
};

/* call when a mission is solved; silent=true re-shows a solved flag without scrolling */
BBM.win = function(n, extraHTML, silent){
  BBM.complete(n);
  const w=document.getElementById("win");
  if(w){
    const L=LEVELS.find(x=>x.n===n);
    w.innerHTML='<h3>🏁 Flag captured</h3><div class="flag">FLAG{'+L.flag+'}</div>'+
      (extraHTML?('<p class="small" style="margin:.5em 0 0">'+extraHTML+'</p>'):'')+
      '<div class="cta"><a class="btn primary" href="'+(n<5?('level-'+(n+1)+'.html'):'victory.html')+'">'+(n<5?('Next: Level '+(n+1)+' →'):'See your results →')+'</a></div>';
    w.classList.add("show");
    if(!silent) w.scrollIntoView({behavior:"smooth",block:"center"});
  }
  const nb=document.getElementById("nextbtn");
  if(nb){ nb.style.pointerEvents=""; nb.style.opacity=""; nb.dataset.locked=""; }
};
/* re-show flag on load if the level is already solved */
BBM.restore = function(n){ if(BBM.done(n)) BBM.win(n,"",true); };

/* ---------------- shared mechanic helpers ---------------------------------- */
// BPE (Level 1)
function getStats(ids){const c=new Map();for(let i=0;i<ids.length-1;i++){const k=ids[i]+","+ids[i+1];c.set(k,(c.get(k)||0)+1);}return c;}
function mergeIds(ids,a,b,idx){const o=[];let i=0;while(i<ids.length){if(i<ids.length-1&&ids[i]===a&&ids[i+1]===b){o.push(idx);i+=2;}else{o.push(ids[i]);i++;}}return o;}
function BPE(){this.merges=new Map();this.vocab=new Map();for(let i=0;i<256;i++)this.vocab.set(i,[i]);}
BPE.prototype.train=function(t,vs){let ids=[...new TextEncoder().encode(t)];for(let n=0;n<vs-256;n++){const st=getStats(ids);if(!st.size)break;let best=null,bc=-1;for(const[k,v]of st)if(v>bc){bc=v;best=k;}const[a,b]=best.split(",").map(Number);const idx=256+n;ids=mergeIds(ids,a,b,idx);this.merges.set(best,idx);this.vocab.set(idx,[...this.vocab.get(a),...this.vocab.get(b)]);}};
BPE.prototype.encode=function(t){let ids=[...new TextEncoder().encode(t)];while(ids.length>=2){const st=getStats(ids);let p=null,r=Infinity;for(const k of st.keys()){const rr=this.merges.has(k)?this.merges.get(k):Infinity;if(rr<r){r=rr;p=k;}}if(p===null||!this.merges.has(p))break;const[a,b]=p.split(",").map(Number);ids=mergeIds(ids,a,b,this.merges.get(p));}return ids;};
BPE.prototype.piece=function(id){return new TextDecoder().decode(new Uint8Array(this.vocab.get(id)));};
BBM.makeBPE = function(){
  const corpus=("the model reads tokens and predicts the next token. a tokenizer turns text into tokens and back. "+
    "the forbidden word is forbidden and remains forbidden here. subword merges compress common byte pairs. ").repeat(30);
  const b=new BPE(); b.train(corpus,320); return b;
};

// vectors (Levels 2)
BBM.nrm=v=>Math.sqrt(v.reduce((s,x)=>s+x*x,0));
BBM.dot=(a,b)=>{let s=0;for(let k=0;k<a.length;k++)s+=a[k]*b[k];return s;};
BBM.cos=(a,b)=>{const na=BBM.nrm(a),nb=BBM.nrm(b);return na&&nb?BBM.dot(a,b)/(na*nb):0;};
if(D.l2){
  const L2=D.l2, dim=L2.E[0].length; BBM.l2dim=dim;
  const widx={}; L2.vocab.forEach((w,i)=>widx[w]=i); BBM.widx=widx;
  BBM.meanVec=ids=>{const m=new Array(dim).fill(0);ids.forEach(id=>{for(let k=0;k<dim;k++)m[k]+=L2.E[id][k];});for(let k=0;k<dim;k++)m[k]/=ids.length;return m;};
  BBM.embed=t=>{const ids=(t.toLowerCase().match(/[a-z]+/g)||[]).map(w=>widx[w]).filter(i=>i!==undefined);return ids.length?BBM.meanVec(ids):new Array(dim).fill(0);};
  BBM.hotflip=function(qvec,nTok,iters){const V=L2.vocab.length;let ids=[];for(let i=0;i<nTok;i++)ids.push(Math.floor(Math.random()*V));
    const sim=s=>BBM.cos(BBM.meanVec(s),qvec);let cur=sim(ids);
    for(let it=0;it<iters;it++){const m=BBM.meanVec(ids),nm=BBM.nrm(m)||1e-9,nq=BBM.nrm(qvec)||1e-9,mq=BBM.dot(m,qvec);
      const g=new Array(dim);for(let k=0;k<dim;k++)g[k]=qvec[k]/(nm*nq)-mq*m[k]/(nm*nm*nm*nq);
      const sc=L2.E.map(r=>BBM.dot(r,g));const cand=[...sc.keys()].sort((a,b)=>sc[b]-sc[a]).slice(0,10);
      let best=null;for(let i=0;i<ids.length;i++)for(const v of cand){if(v===ids[i])continue;const tr=ids.slice();tr[i]=v;const s=sim(tr);if(!best||s>best.s)best={s,i,v};}
      if(!best||best.s<=cur+1e-9)break;cur=best.s;ids[best.i]=best.v;}
    return {words:ids.map(i=>L2.vocab[i]),sim:cur};};
}

// softmax (Level 4)
BBM.softmax=z=>{const mx=Math.max(...z),e=z.map(x=>Math.exp(x-mx)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/s);};
})();
