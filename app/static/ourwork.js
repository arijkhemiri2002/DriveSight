/* DriveSight - "Our work" panel, shared by the photo and video pages.
   Opens from any element with data-ourwork. Figures come from the project README (V2 validation split,
   686 images) and from reports/video_dataset (phase 2-5 completion reports, validation split).
   Edit them here if the results change. */
(() => {
const MODELS = [
  {name:"YOLO11n", role:"reference", params:"2.6 M", map:29.43, map50:54.74, p:70.5, r:48.4, vru:20.50, vrur:38.8, ms:19.2},
  {name:"YOLOv8s", role:"selected", params:"11.1 M", map:31.51, map50:57.58, p:71.8, r:51.9, vru:23.88, vrur:43.8, ms:31.1},
  {name:"YOLO11s", role:"best recall", params:"9.4 M", map:31.54, map50:57.48, p:66.5, r:53.0, vru:21.55, vrur:46.2, ms:31.1},
];
const TEAM = ["Arij Khemiri", "Kenza Yahiaoui", "Wahbi Jouini", "Rayen"];
// Phase 5, validation split (95 clips), same-time AUC: 0.5 = chance, and = what clip time alone gets
const RISK = [
  {name:"Full model", sub:"tracks + gaze + GPS", v:0.744, lo:0.664, hi:0.817, sel:true},
  {name:"Speed only", sub:"reference", v:0.730},
  {name:"Vision only", sub:"used in this app", v:0.628, lo:0.557, hi:0.700, app:true},
  {name:"Fixed rule", sub:"baseline, not trained", v:0.578, lo:0.478, hi:0.682},
  {name:"Clip time only", sub:"the shortcut", v:0.500, ref:true},
];
const ABL = [["Objects", .593], ["+ motion", .628], ["+ attention", .635], ["GPS only", .741], ["Everything", .744]];

const css = `
.ow-back{position:fixed;inset:0;z-index:30;background:rgba(3,4,10,.62);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);
  display:none;align-items:flex-start;justify-content:center;overflow:auto;padding:28px 16px}
.ow-back.on{display:flex}
.ow{position:relative;width:min(1080px,100%);padding:34px 34px 28px;margin:auto 0}
.ow h2{font-size:30px;font-weight:750;letter-spacing:-.4px;max-width:24ch;line-height:1.15}
.ow .lead{color:var(--mute);font-size:15px;line-height:1.6;max-width:68ch;margin-top:12px}
.ow h3{font-size:15px;font-weight:700;margin:30px 0 14px;color:var(--text)}
.ow .x{position:absolute;top:18px;right:18px}
.ow-steps{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;counter-reset:s}
.ow-step{position:relative;background:rgba(255,255,255,.045);border:1px solid var(--edge);border-radius:16px;padding:16px 16px 14px}
.ow-step::before{counter-increment:s;content:counter(s);position:absolute;top:14px;right:14px;width:24px;height:24px;border-radius:50%;
  display:grid;place-items:center;font-size:12px;font-weight:700;color:#fff;background:linear-gradient(135deg,var(--accent),var(--accent2))}
.ow-step b{display:block;font-size:14.5px;margin-bottom:6px;padding-right:30px}
.ow-step p{font-size:12.8px;color:var(--mute);line-height:1.55}
.ow-step .k{display:block;font-size:24px;font-weight:750;margin-top:10px;background:linear-gradient(135deg,#c9d0ff,#e3c9ff);-webkit-background-clip:text;background-clip:text;color:transparent}
.ow-step .k small{font-size:12px;font-weight:600;color:var(--mute);-webkit-text-fill-color:var(--mute);margin-left:4px}
.ow-table{width:100%;border-collapse:separate;border-spacing:0;font-size:13.5px}
.ow-table th{font-weight:600;color:var(--mute);text-align:right;padding:9px 12px;border-bottom:1px solid var(--edge);white-space:nowrap}
.ow-table th:first-child,.ow-table td:first-child{text-align:left}
.ow-table td{text-align:right;padding:11px 12px;border-bottom:1px solid var(--edge);font-variant-numeric:tabular-nums}
.ow-table tr.sel td{background:rgba(var(--accent-rgb),.12)}
.ow-table tr.sel td:first-child{border-radius:12px 0 0 12px} .ow-table tr.sel td:last-child{border-radius:0 12px 12px 0}
.ow-table .best{color:#fff;font-weight:700}
.ow-tag{display:inline-block;font-size:11px;font-weight:700;padding:3px 8px;border-radius:99px;margin-left:8px;vertical-align:1px;color:var(--mute);border:1px solid var(--edge2)}
.ow-tag.sel{color:#fff;border:0;background:linear-gradient(135deg,var(--accent),var(--accent2))}
.ow-bars{display:flex;flex-direction:column;gap:10px;margin-top:6px}
.ow-bar{display:grid;grid-template-columns:90px 1fr 54px;gap:12px;align-items:center;font-size:13px}
.ow-bar i{display:block;height:12px;border-radius:6px;background:rgba(255,255,255,.08);position:relative;overflow:hidden}
.ow-bar i u{position:absolute;inset:0 auto 0 0;border-radius:6px;background:rgba(255,255,255,.28)}
.ow-bar.sel i u{background:linear-gradient(90deg,var(--accent),var(--accent2));box-shadow:0 0 14px rgba(var(--accent-rgb),.5)}
.ow-bar b{text-align:right;font-variant-numeric:tabular-nums}
.ow-two{display:grid;grid-template-columns:1.15fr 1fr;gap:26px}
.ow-note{font-size:12.5px;color:var(--dim);line-height:1.55;margin-top:12px}
.ow-foot{display:flex;flex-wrap:wrap;gap:22px;justify-content:space-between;margin-top:30px;padding-top:18px;border-top:1px solid var(--edge);font-size:12.5px;color:var(--mute);line-height:1.6}
.ow-foot b{color:var(--text)}
.ow-team{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}
.ow-team span{padding:6px 12px;border-radius:99px;background:rgba(255,255,255,.06);border:1px solid var(--edge);color:var(--text);font-weight:600}
.ow-steps.five{grid-template-columns:repeat(5,1fr)}
.ow-new{position:absolute;left:14px;top:-9px;font-size:10.5px;font-weight:800;letter-spacing:.3px;padding:2px 8px;border-radius:99px;color:#0a1020;
  background:linear-gradient(135deg,#7cf5c8,#c9f77c)}
.ow-step.new{border-color:rgba(124,245,200,.45);background:rgba(124,245,200,.06)}
.ow-step .who{display:block;font-size:11.5px;color:var(--dim);margin-top:8px}
.ow-auc{display:flex;flex-direction:column;gap:12px;margin-top:4px}
.ow-two>*{min-width:0}
.ow-auc .r{display:grid;grid-template-columns:150px minmax(0,1fr) 48px;gap:14px;align-items:center}
.ow-auc .n{font-size:13.5px;font-weight:650;line-height:1.2}
.ow-auc .n small{display:block;font-size:11.5px;font-weight:500;color:var(--mute)}
.ow-auc .t{position:relative;height:14px;border-radius:7px;background:rgba(255,255,255,.06)}
.ow-auc .t u{position:absolute;inset:0 auto 0 0;border-radius:7px;background:rgba(255,255,255,.28)}
.ow-auc .r.sel u{background:linear-gradient(90deg,var(--accent),var(--accent2));box-shadow:0 0 14px rgba(var(--accent-rgb),.5)}
.ow-auc .r.in-app u{background:rgba(124,245,200,.65)}
.ow-auc .r.ref u{background:repeating-linear-gradient(135deg,rgba(255,255,255,.18) 0 6px,rgba(255,255,255,.06) 6px 12px)}
.ow-auc .t s{position:absolute;top:-4px;bottom:-4px;border-left:2px solid rgba(238,243,255,.85);border-right:2px solid rgba(238,243,255,.85);
  border-radius:3px;text-decoration:none;opacity:.55}
.ow-auc .t i{position:absolute;top:-8px;bottom:-8px;width:0;border-left:1.5px dashed rgba(238,243,255,.5)}
.ow-auc b{text-align:right;font-variant-numeric:tabular-nums}
.ow-axis{display:grid;grid-template-columns:150px minmax(0,1fr) 48px;gap:14px;font-size:11px;color:var(--dim)}
.ow-axis div{display:flex;justify-content:space-between}
.ow-mini{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
.ow-mini span{font-size:12px;padding:6px 10px;border-radius:10px;background:rgba(255,255,255,.05);border:1px solid var(--edge);color:var(--mute)}
.ow-mini span b{color:var(--text);margin-left:6px;font-variant-numeric:tabular-nums}
.ow-traps{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.ow-trap{border-radius:16px;padding:16px;background:rgba(255,181,71,.06);border:1px solid rgba(255,181,71,.28)}
.ow-trap b{display:block;font-size:14.5px;margin-bottom:6px}
.ow-trap p{font-size:12.8px;color:var(--mute);line-height:1.55}
.ow-trap em{display:block;font-style:normal;font-size:12.5px;color:#7cf5c8;margin-top:10px;line-height:1.45}
.ow-checks{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.ow-check{padding:14px 16px;border-radius:14px;background:rgba(124,245,200,.05);border:1px solid rgba(124,245,200,.25)}
.ow-check b{display:block;font-size:20px;font-weight:750;color:#7cf5c8;font-variant-numeric:tabular-nums}
.ow-check span{font-size:12.5px;color:var(--mute);line-height:1.45}
@media (max-width:900px){.ow-steps.five,.ow-traps,.ow-checks{grid-template-columns:1fr 1fr}.ow-auc .r,.ow-axis{grid-template-columns:110px minmax(0,1fr) 40px}}
@media (max-width:520px){.ow-steps.five,.ow-traps,.ow-checks{grid-template-columns:1fr}}
@media (max-width:900px){.ow-steps{grid-template-columns:1fr 1fr}.ow-two{grid-template-columns:1fr}.ow{padding:26px 18px 22px}.ow h2{font-size:24px}
  .ow-table{font-size:12px}.ow-table th,.ow-table td{padding:8px 6px}.ow-hide{display:none}}
@media (max-width:520px){.ow-steps{grid-template-columns:1fr}}
`;

const best = k => Math.max(...MODELS.map(m=>m[k]));
const minv = k => Math.min(...MODELS.map(m=>m[k]));
const cell = (m,k,fmt,low) => `<td class="${m[k]===(low?minv(k):best(k))?"best":""}">${fmt(m[k])}</td>`;
const f1 = v => v.toFixed(1), f2 = v => v.toFixed(2);
const bars = (k, max) => MODELS.map(m=>`<div class="ow-bar ${m.role==="selected"?"sel":""}"><span>${m.name}</span><i><u style="width:${(m[k]/max*100).toFixed(1)}%"></u></i><b>${f1(m[k])}</b></div>`).join("");

const html = `
<div class="ow glass" role="dialog" aria-modal="true" aria-labelledby="ow-title">
  <button class="btn icon x" type="button" data-ow-close aria-label="Close"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M6 6l12 12M18 6L6 18"/></svg></button>
  <h2 id="ow-title">Seeing road users, following them, and predicting the brake</h2>
  <p class="lead">DriveSight runs in five stages on Berkeley DeepDrive data. A detector tuned for pedestrians and cyclists finds road users;
    a tracker follows each one over time; a risk model combines their motion with where attentive drivers look and with the car's speed,
    and estimates five times a second whether a hard brake is coming within 2 seconds.</p>

  <h3>How we built it</h3>
  <div class="ow-steps five">
    <div class="ow-step"><b>Detector</b><p>Three YOLO models trained on BDD100K with weighting toward rare classes. YOLOv8s wins on vulnerable road users.</p><span class="k num">23.9<small>VRU AP</small></span><span class="who">Arij</span></div>
    <div class="ow-step"><b>Drive data</b><p>1,161 BDDA drives audited and cut into 200 ms samples, each with the attention map and the GPS, nothing stored twice.</p><span class="k num">62,008<small>samples</small></span><span class="who">Arij</span></div>
    <div class="ow-step"><b>Detection</b><p>YOLOv8s run once on every sample, low-confidence boxes kept on purpose for the tracker.</p><span class="k num">2.5 M<small>boxes</small></span><span class="who">Arij</span></div>
    <div class="ow-step new"><span class="ow-new">NEW</span><b>Tracking</b><p>ByteTrack gives every road user a fixed number from frame to frame, tuned on the validation drives only.</p><span class="k num">47,221<small>tracks</small></span><span class="who">Kenza</span></div>
    <div class="ow-step new"><span class="ow-new">NEW</span><b>Fusion and risk</b><p>45 measures per sample: who is in our lane, how fast it grows, time to contact, attention, speed. Then a learned risk score.</p><span class="k num">0.744<small>same-time AUC</small></span><span class="who">Kenza</span></div>
  </div>

  <div class="ow-two">
    <div>
      <h3>Detector comparison <span class="ow-tag">validation, 686 images</span></h3>
      <table class="ow-table">
        <thead><tr><th>Model</th><th>mAP50-95</th><th>VRU AP50-95</th><th>VRU recall</th><th class="ow-hide">Precision</th><th>ms / image</th></tr></thead>
        <tbody>${MODELS.map(m=>`<tr class="${m.role==="selected"?"sel":""}"><td><b>${m.name}</b><span class="ow-tag ${m.role==="selected"?"sel":""}">${m.role}</span></td>
          ${cell(m,"map",f2)}${cell(m,"vru",f2)}${cell(m,"vrur",f1)}<td class="ow-hide ${m.p===best("p")?"best":""}">${f1(m.p)}</td>${cell(m,"ms",f1,true)}</tr>`).join("")}</tbody>
      </table>
      <p class="ow-note">VRU = pedestrian, rider, motorcycle, bicycle. Only YOLOv8s beats the YOLO11n reference on VRU AP50-95 with a 95% interval that excludes zero (+3.4, [+0.8, +6.3]). Latency: batch 1, 1152 px, RTX 2050.</p>
    </div>
    <div>
      <h3>Following every road user <span class="ow-tag sel">new</span></h3>
      <div class="ow-mini" style="margin-top:0">
        <span>Confident boxes kept in a track<b>84 %</b></span><span>Median track<b>2.2 s</b></span>
        <span>Vehicles covered<b>92 %</b></span><span>Pedestrians and cyclists<b>66 %</b></span>
      </div>
      <h3>Broken tracks, before and after tuning</h3>
      <div class="ow-bars">
        <div class="ow-bar"><span>Default</span><i><u style="width:80%"></u></i><b>24 %</b></div>
        <div class="ow-bar sel"><span>Tuned</span><i><u style="width:35%"></u></i><b>11 %</b></div>
      </div>
      <p class="ow-note">Share of tracks lasting only 1 or 2 frames, out of 81 settings tried on the 95 validation drives. The same figures hold on the 800 training drives,
        which were not used for tuning. Pedestrians are harder to follow: small, fast, and seen 5 times a second.</p>
    </div>
  </div>

  <h3>Predicting a hard brake <span class="ow-tag sel">new</span></h3>
  <div class="ow-two">
    <div>
      <div class="ow-auc">
        ${RISK.map(m=>`<div class="r ${m.sel?"sel":""} ${m.app?"in-app":""} ${m.ref?"ref":""}"><div class="n">${m.name}<small>${m.sub}</small></div>
          <div class="t"><i style="left:${((0.5-0.4)/0.45*100).toFixed(1)}%"></i><u style="width:${((m.v-0.4)/0.45*100).toFixed(1)}%"></u>
          ${m.lo?`<s style="left:${((m.lo-0.4)/0.45*100).toFixed(1)}%;width:${((m.hi-m.lo)/0.45*100).toFixed(1)}%"></s>`:""}</div><b>${m.v.toFixed(3)}</b></div>`).join("")}
        <div class="ow-axis"><span></span><div><span>0.40</span><span>0.50 chance</span><span>0.85</span></div><span></span></div>
      </div>
      <p class="ow-note">Same-time AUC on the 95 validation drives, with 95 % intervals from resampling whole drives. It only compares moments taken at the same second
        of different clips, so a model cannot win by knowing how far into the clip it is.</p>
    </div>
    <div>
      <div class="ow-steps" style="grid-template-columns:1fr;gap:10px">
        <div class="ow-step"><b>Label without annotation</b><p>A hard brake is a GPS deceleration of 2 m/s² or more. The model must flag it up to 2 s ahead, using only the past.</p></div>
        <div class="ow-step"><b>What each source brings</b><p>Most of the signal is in the speed and a slowdown that has already begun. Above 25 km/h, speed alone is at chance (0.495)
          while the full model keeps 0.614: it uses more than speed.</p></div>
      </div>
      <div class="ow-mini">${ABL.map(([n,v])=>`<span>${n}<b>${v.toFixed(3)}</b></span>`).join("")}</div>
    </div>
  </div>

  <h3>Traps we caught</h3>
  <div class="ow-traps">
    <div class="ow-trap"><b>The clip gives the answer away</b><p>In BDDA, two clips out of three start braking at exactly 5.5 s. Clip time alone looks like a good model.</p>
      <em>Time is never an input, and every score is compared at the same second of the clip.</em></div>
    <div class="ow-trap"><b>A hidden leak</b><p>"Seconds since the last GPS fix" revealed the time inside each second, enough on its own for an AUC of 0.61.</p>
      <em>Found, removed from the inputs, and guarded by a test.</em></div>
    <div class="ow-trap"><b>Speed explains a lot</b><p>A car crawling along cannot brake hard, so speed alone scores 0.730.</p>
      <em>Reported next to the model, with results split by speed.</em></div>
  </div>

  <h3>Checked, not assumed</h3>
  <div class="ow-checks">
    <div class="ow-check"><b>4 / 4</b><span>independent verifiers report "all checks passed" (phases 2 to 5)</span></div>
    <div class="ow-check"><b>57</b><span>automated tests, including a fake leak that the verifier must catch</span></div>
    <div class="ow-check"><b>0</b><span>features that change when the future is cut off (tested on 45 drives)</span></div>
    <div class="ow-check"><b>2 PCs</b><span>same tracks and same 45 features, value for value</span></div>
  </div>
  <p class="ow-note">All figures above are on the validation split. The 266 test drives are kept untouched for one final evaluation.
    The video page uses the vision part of the model, because a plain video has no attention map and no GPS.</p>

  <div class="ow-foot">
    <div><b>Team</b><div class="ow-team">${TEAM.map(n=>`<span>${n}</span>`).join("")}</div></div>
    <div style="max-width:52ch"><b>Credits.</b> BDD100K and BDD-A (Berkeley DeepDrive), research and education use only. Detectors by Ultralytics (AGPL-3.0).
      A research demo, not a safety system.</div>
  </div>
</div>`;

function init(){
  const st = document.createElement("style"); st.textContent = css; document.head.appendChild(st);
  const back = document.createElement("div"); back.className = "ow-back"; back.innerHTML = html; document.body.appendChild(back);
  let last = null;
  const open = () => { last = document.activeElement; back.classList.add("on"); back.querySelector("[data-ow-close]").focus(); };
  const close = () => { back.classList.remove("on"); last && last.focus && last.focus(); };
  document.querySelectorAll("[data-ourwork]").forEach(b=>b.addEventListener("click", e=>{ e.preventDefault(); e.stopPropagation(); open(); }));
  back.addEventListener("click", e=>{ if(e.target===back || e.target.closest("[data-ow-close]")) close(); });
  document.addEventListener("keydown", e=>{ if(e.key==="Escape" && back.classList.contains("on")) close(); }, true);
  window.openOurWork = open;
}
document.readyState === "loading" ? document.addEventListener("DOMContentLoaded", init) : init();
})();
