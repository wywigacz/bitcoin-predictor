(function(){
const D=window.__BTC__, N=D.dates.length, LAST=D.dates[N-1], MAXH=Math.max(...D.fc.map(f=>f.h));
const css=getComputedStyle(document.documentElement), C=n=>css.getPropertyValue(n).trim();
const fmtD=d=>new Date(d*864e5).toISOString().slice(0,10);
const fmtP=v=>v==null?'–':Math.round(v).toLocaleString('pl-PL').replace(/\s/g,' ');
const el=document.getElementById('chart'), cx=document.getElementById('cx'), tip=document.getElementById('tip');
let months=12,showEv=true;
const evBox=document.getElementById('evtoggle');if(evBox)evBox.onchange=()=>{showEv=evBox.checked;draw();};
const keys=['p5','p10','p25','p50','p75','p90','p95'];
const fcPts=k=>[[LAST,D.price[N-1]]].concat(D.fc.map(f=>[LAST+f.h,f[k]]));
function interp(k,x){const p=fcPts(k);for(let i=1;i<p.length;i++){if(x<=p[i][0]){const t=(x-p[i-1][0])/(p[i][0]-p[i-1][0]);return Math.exp(Math.log(p[i-1][1])+t*(Math.log(p[i][1])-Math.log(p[i-1][1])));}}return p[p.length-1][1];}
function niceLog(lo,hi){const out=[];const a=Math.floor(Math.log10(lo)),b=Math.ceil(Math.log10(hi));
 for(let e=a;e<=b;e++)for(const m of [1,2,5]){const v=m*Math.pow(10,e);if(v>=lo&&v<=hi)out.push(v);}return out;}
let geo={};
function draw(){
 const W=el.clientWidth||900,H=Math.round(Math.max(300,Math.min(460,W*.42))),pl=58,pr=16,pt=12,pb=26;
 const x0=months?LAST-months*30.44:D.dates[0], x1=LAST+MAXH+Math.max(6,months?months*1.2:60);
 let i0=0;while(i0<N&&D.dates[i0]<x0)i0++;
 let lo=Infinity,hi=-Infinity;
 const upd=v=>{if(v!=null&&v>0){if(v<lo)lo=v;if(v>hi)hi=v;}};
 for(let i=i0;i<N;i++){upd(D.price[i]);upd(D.real[i]);upd(D.ma200w[i]);}
 D.fc.forEach(f=>{upd(f.p5);upd(f.p95);});
 lo*=.93;hi*=1.07;const L0=Math.log(lo),L1=Math.log(hi);
 const X=x=>pl+(x-x0)/(x1-x0)*(W-pl-pr),Y=v=>pt+(1-(Math.log(v)-L0)/(L1-L0))*(H-pt-pb);
 geo={x0,x1,pl,pr,W,H,X,Y,i0};
 let s=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"><defs>
 <linearGradient id="af" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${C('--accent')}" stop-opacity=".30"/><stop offset="1" stop-color="${C('--accent')}" stop-opacity="0"/></linearGradient>
 <linearGradient id="bf" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="${C('--blue')}" stop-opacity=".10"/><stop offset="1" stop-color="${C('--blue')}" stop-opacity=".34"/></linearGradient>
 <linearGradient id="bf2" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="${C('--blue')}" stop-opacity=".22"/><stop offset="1" stop-color="${C('--blue')}" stop-opacity=".58"/></linearGradient>
 <clipPath id="cp"><rect x="${pl}" y="${pt}" width="${W-pl-pr}" height="${H-pt-pb}"/></clipPath></defs>`;
 niceLog(lo,hi).forEach(v=>{s+=`<line x1="${pl}" x2="${W-pr}" y1="${Y(v)}" y2="${Y(v)}" class="gridl"/><text x="${pl-8}" y="${Y(v)+4}" text-anchor="end" class="ax">${v>=1000?(v/1000)+'k':v}</text>`;});
 const span=(x1-x0)/30.44, step=span>60?24:span>30?12:span>14?6:span>6?3:1;
 const d0=new Date(x0*864e5);let mo=new Date(Date.UTC(d0.getUTCFullYear(),d0.getUTCMonth()+1,1));
 while(mo.getTime()/864e5<x1){const m=mo.getUTCMonth();const x=mo.getTime()/864e5;
  if((mo.getUTCFullYear()*12+m)%step===0){s+=`<text x="${X(x)}" y="${H-7}" text-anchor="middle" class="ax">${step>=12?mo.getUTCFullYear():mo.toISOString().slice(0,7)}</text>`;}
  mo=new Date(Date.UTC(mo.getUTCFullYear(),m+1,1));}
 s+='<g clip-path="url(#cp)">';
 D.halvings.forEach(h=>{if(h>=x0&&h<=x1)s+=`<line x1="${X(h)}" x2="${X(h)}" y1="${pt}" y2="${H-pb}" class="hv"/><text x="${X(h)+5}" y="${pt+11}" class="hvl">halving</text>`;});
 const band=(a,b,fill)=>{const A=fcPts(a),B=fcPts(b).slice().reverse();s+=`<polygon points="${A.concat(B).map(p=>X(p[0]).toFixed(1)+','+Y(p[1]).toFixed(1)).join(' ')}" fill="${fill}"/>`;};
 band('p5','p95','url(#bf)');band('p25','p75','url(#bf2)');
 const line=(arr,col,w,dash)=>{let d='',pen=false;for(let i=i0;i<N;i++){const v=arr[i];if(v==null){pen=false;continue;}d+=(pen?'L':'M')+X(D.dates[i]).toFixed(1)+' '+Y(v).toFixed(1);pen=true;}
  s+=`<path d="${d}" fill="none" stroke="${col}" stroke-width="${w}" ${dash?`stroke-dasharray="${dash}"`:''} stroke-linejoin="round"/>`;};
 line(D.ma200w,C('--violet'),1.4,'5 4');line(D.real,C('--amber'),1.4,'5 4');
 let ad='M'+X(D.dates[i0]).toFixed(1)+' '+(H-pb);for(let i=i0;i<N;i++)ad+='L'+X(D.dates[i]).toFixed(1)+' '+Y(D.price[i]).toFixed(1);ad+='L'+X(LAST).toFixed(1)+' '+(H-pb)+'Z';
 s+=`<path d="${ad}" fill="url(#af)"/>`;
 line(D.price,C('--fg'),1.7);
 geo.ev=[];
 if(showEv)D.events.forEach(e=>{if(e.d<x0||e.d>LAST)return;const p=D.price[e.d-D.dates[0]];if(p==null)return;
  const col=e.r==null?C('--muted'):(e.r>=0?C('--green'):C('--red'));geo.ev.push({x:X(e.d),y:Y(p),e});
  s+=`<circle cx="${X(e.d)}" cy="${Y(p)}" r="4" fill="${col}" stroke="${C('--bg2')}" stroke-width="1.5"/>`;});
 s+=`<path d="${fcPts('p50').map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join('')}" fill="none" stroke="${C('--accent')}" stroke-width="2.2" stroke-dasharray="6 4"/>`;
 s+=`<line x1="${X(LAST)}" x2="${X(LAST)}" y1="${pt}" y2="${H-pb}" class="hv"/>`;
 D.fc.forEach(f=>{s+=`<circle cx="${X(LAST+f.h)}" cy="${Y(f.p50)}" r="4" fill="${C('--accent')}" stroke="${C('--bg2')}" stroke-width="2"/>`;});
 s+=`<circle cx="${X(LAST)}" cy="${Y(D.price[N-1])}" r="5" fill="${C('--fg')}" stroke="${C('--bg2')}" stroke-width="2"/></g></svg>`;
 el.querySelector('svg')?.remove();el.insertAdjacentHTML('afterbegin',s);
}
function hover(e){
 const r=el.getBoundingClientRect(),px=e.clientX-r.left;if(px<geo.pl||px>geo.W-geo.pr){off();return;}
 const x=geo.x0+(px-geo.pl)/(geo.W-geo.pl-geo.pr)*(geo.x1-geo.x0);let html;
 const near=(geo.ev||[]).reduce((b,o)=>Math.abs(o.x-px)<9&&(!b||Math.abs(o.x-px)<Math.abs(b.x-px))?o:b,null);
 if(near){const e=near.e;html=`<b>${fmtD(e.d)}</b><div style="display:block;color:var(--fg);margin-bottom:4px">${e.t}</div><div><span>Kategoria</span><span>${e.c}</span></div>`+(e.r==null?'':`<div><span>BTC po 3 dniach</span><span style="color:${e.r>=0?C('--green'):C('--red')}">${e.r>0?'+':''}${e.r}%</span></div>`);}
 else if(x<=LAST+.5){let lo=0,hi=N-1;while(lo<hi){const m=(lo+hi)>>1;if(D.dates[m]<x)lo=m+1;else hi=m;}
  const i=Math.max(0,lo-(Math.abs(D.dates[lo-1]-x)<Math.abs(D.dates[lo]-x)?1:0));
  html=`<b>${fmtD(D.dates[i])}</b><div><span>Cena</span><span>${fmtP(D.price[i])} USD</span></div>`+(D.real[i]!=null?`<div><span>Średnia cena zakupu</span><span>${fmtP(D.real[i])}</span></div>`:'')+(D.ma200w[i]!=null?`<div><span>Średnia z 4 lat</span><span>${fmtP(D.ma200w[i])}</span></div>`:'');
 }else{const xs=Math.min(x,LAST+MAXH);html=`<b>${fmtD(Math.round(x))} <span style="color:var(--accent)">prognoza</span></b>`+
  [['P90','p90'],['Mediana','p50'],['P10','p10']].map(([a,k])=>`<div><span>${a}</span><span>${fmtP(interp(k,xs))} USD</span></div>`).join('');}
 cx.style.display='block';cx.style.left=px+'px';tip.innerHTML=html;tip.style.display='block';
 const tw=tip.offsetWidth;tip.style.left=Math.min(Math.max(px+14,0),r.width-tw)+'px';tip.style.top='12px';}
function off(){cx.style.display='none';tip.style.display='none';}
el.addEventListener('pointermove',hover);el.addEventListener('pointerleave',off);
document.querySelectorAll('#rng button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#rng button').forEach(x=>x.classList.remove('on'));b.classList.add('on');months=+b.dataset.m;draw();});

/* wykres wskaźnika złożonego */
const el2=document.getElementById('chart2'),tip2=document.getElementById('tip2');let g2={};
function draw2(){
 const W=el2.clientWidth||900,H=Math.round(Math.max(190,Math.min(260,W*.24))),pl=40,pr=16,pt=10,pb=24;
 let i0=0;while(i0<N&&D.comp[i0]==null)i0++;const x0=D.dates[i0],x1=LAST;
 const X=x=>pl+(x-x0)/(x1-x0)*(W-pl-pr),Y=v=>pt+(1-(v+1)/2)*(H-pt-pb);g2={x0,x1,pl,pr,W,H};
 let s=`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`;
 [-1,-.5,0,.5,1].forEach(v=>{s+=`<line x1="${pl}" x2="${W-pr}" y1="${Y(v)}" y2="${Y(v)}" class="gridl" ${v===0?`style="stroke:var(--line2)"`:''}/><text x="${pl-8}" y="${Y(v)+4}" text-anchor="end" class="ax">${v>0?'+':''}${v}</text>`;});
 for(let y=new Date(x0*864e5).getUTCFullYear()+1;y<=new Date(x1*864e5).getUTCFullYear();y++){const d=Date.UTC(y,0,1)/864e5;if(y%2===0)s+=`<text x="${X(d)}" y="${H-7}" text-anchor="middle" class="ax">${y}</text>`;}
 D.halvings.forEach(h=>{if(h>=x0)s+=`<line x1="${X(h)}" x2="${X(h)}" y1="${pt}" y2="${H-pb}" class="hv"/><text x="${X(h)+5}" y="${pt+11}" class="hvl">halving</text>`;});
 const area=(f,col)=>{let d='M'+X(D.dates[i0])+' '+Y(0),last=null;for(let i=i0;i<N;i++){const v=D.comp[i];if(v==null)continue;d+='L'+X(D.dates[i]).toFixed(1)+' '+Y(f(v)).toFixed(1);last=i;}
  d+='L'+X(D.dates[last]).toFixed(1)+' '+Y(0)+'Z';s+=`<path d="${d}" fill="${col}" fill-opacity=".34"/>`;};
 area(v=>Math.max(v,0),C('--green'));area(v=>Math.min(v,0),C('--red'));
 let d='',pen=false;for(let i=i0;i<N;i++){const v=D.comp[i];if(v==null){pen=false;continue;}d+=(pen?'L':'M')+X(D.dates[i]).toFixed(1)+' '+Y(v).toFixed(1);pen=true;}
 s+=`<path d="${d}" fill="none" stroke="${C('--fg')}" stroke-width="1.3"/></svg>`;
 el2.querySelector('svg')?.remove();el2.insertAdjacentHTML('afterbegin',s);}
el2.addEventListener('pointermove',e=>{const r=el2.getBoundingClientRect(),px=e.clientX-r.left;if(px<g2.pl||px>g2.W-g2.pr){tip2.style.display='none';return;}
 const x=g2.x0+(px-g2.pl)/(g2.W-g2.pl-g2.pr)*(g2.x1-g2.x0);let i=Math.round(N-1-(LAST-x));i=Math.min(N-1,Math.max(0,i));const v=D.comp[i];
 if(v==null){tip2.style.display='none';return;}tip2.innerHTML=`<b>${fmtD(D.dates[i])}</b><div><span>Wskaźnik</span><span>${v>0?'+':''}${v.toFixed(2)}</span></div><div><span>Cena</span><span>${fmtP(D.price[i])} USD</span></div>`;
 tip2.style.display='block';tip2.style.left=Math.min(Math.max(px+14,0),r.width-tip2.offsetWidth)+'px';tip2.style.top='8px';});
el2.addEventListener('pointerleave',()=>tip2.style.display='none');
function all(){draw();draw2();}
window.__redraw=all;all();let rt;addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(all,120);});
})();
