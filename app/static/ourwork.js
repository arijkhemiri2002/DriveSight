/* DriveSight - "Our work" panel, shared by the photo and video pages.
   Opens from any element with data-ourwork. Figures come from the project README (V2 validation split,
   686 images) and from reports/video_dataset (BDDA audit). Edit them here if the results change. */
(() => {
const MODELS = [
  {name:"YOLO11n", role:"reference", params:"2.6 M", map:29.43, map50:54.74, p:70.5, r:48.4, vru:20.50, vrur:38.8, ms:19.2},
  {name:"YOLOv8s", role:"selected", params:"11.1 M", map:31.51, map50:57.58, p:71.8, r:51.9, vru:23.88, vrur:43.8, ms:31.1},
  {name:"YOLO11s", role:"best recall", params:"9.4 M", map:31.54, map50:57.48, p:66.5, r:53.0, vru:21.55, vrur:46.2, ms:31.1},
];
const TEAM = ["Arij Khemiri", "Kenza Yahiaoui", "Wahbi Jouini"];

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
  <h2 id="ow-title">Seeing pedestrians and cyclists first, then predicting the brake</h2>
  <p class="lead">DriveSight is a two-stage system built on Berkeley DeepDrive data. A road-object detector, tuned for vulnerable road users,
    feeds a risk model that estimates, five times a second, whether the driver is about to brake hard.</p>

  <h3>How we built it</h3>
  <div class="ow-steps">
    <div class="ow-step"><b>Curated the data</b><p>BDD100K driving images converted to 9 road classes, with a protected test split never used for model choice.</p><span class="k num">4,260<small>images</small></span></div>
    <div class="ow-step"><b>Trained for vulnerable users</b><p>Class re-weighting toward rare classes and 1152 px inputs to catch small pedestrians, on a 4 GB laptop GPU.</p><span class="k num">1152<small>px input</small></span></div>
    <div class="ow-step"><b>Compared three detectors</b><p>Same data, same recipe, same evaluation, with paired bootstrap confidence intervals. YOLOv8s wins on vulnerable road users.</p><span class="k num">+3.4<small>VRU AP vs YOLO11n</small></span></div>
    <div class="ow-step"><b>Built the risk stage</b><p>BDDA drives audited for camera, gaze and GPS sync; GPS deceleration gives a braking label with no manual annotation.</p><span class="k num">1,161<small>clean drives</small></span></div>
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
      <h3>Average precision on vulnerable road users</h3>
      <div class="ow-bars">${bars("vru", 30)}</div>
      <h3>Recall on vulnerable road users</h3>
      <div class="ow-bars">${bars("vrur", 60)}</div>
    </div>
  </div>

  <h3>The risk stage</h3>
  <div class="ow-steps" style="grid-template-columns:repeat(3,1fr)">
    <div class="ow-step"><b>Label</b><p>Hard braking (GPS deceleration of 2 m/s² or more) within the next 2 seconds. Frames after the brake are left out.</p></div>
    <div class="ow-step"><b>Features</b><p>Road users and their distance to the path, how fast they get closer, where an attentive driver looks (BDDA gaze maps), and speed. Never the future.</p></div>
    <div class="ow-step"><b>Models</b><p>A rule-based score as the baseline against logistic regression and gradient boosting, judged on a time-matched AUC so the clip timing cannot be exploited.</p></div>
  </div>
  <p class="ow-note">Risk-model results on the full BDDA set are in progress. Until the learned model is trained, the video page uses the rule-based score.</p>

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
