async function fetchJSON(url) {
  const r = await fetch(url);
  return await r.json();
}

function byId(x){ return document.getElementById(x); }

function fmt(n){
  if (n === null || n === undefined) return "null";
  if (typeof n === "number") return Number.isFinite(n) ? n.toFixed(6) : String(n);
  return String(n);
}

function isRTK(p){
  return (p.gps_status || "").toUpperCase() === "RTK";
}

function renderItem(p){
  const div = document.createElement("div");
  div.className = "item";
  const yaw = p.gimbal_yaw ?? p.flight_yaw;
  div.innerHTML = `
    <div><b>${p.image_id}</b></div>
    <div>RTK: ${p.gps_status ?? "-" } | yaw=${fmt(yaw)} pitch=${fmt(p.gimbal_pitch)}</div>
    <div style="font-size:12px;color:#555;word-break:break-all">${p.rgb_path}</div>
  `;
  return div;
}

async function main(){
  const poses = await fetchJSON("/api/poses");
  const list = byId("list");
  const rgbImg = byId("rgbImg");
  const tImg = byId("tImg");
  const meta = byId("meta");

  function applyFilter(){
    list.innerHTML = "";
    const onlyRTK = byId("onlyRTK").checked;
    const yawTol = parseFloat(byId("yawTol").value || "9999");

    // 简单示例：只显示 yaw 接近众数的（帮助你快速看同一立面）
    const yaws = poses.map(p => p.gimbal_yaw).filter(v => typeof v === "number");
    const yawRef = yaws.length ? yaws[Math.floor(yaws.length / 2)] : 0;

    poses.forEach(p => {
      if (onlyRTK && !isRTK(p)) return;
      const yaw = p.gimbal_yaw;
      if (typeof yaw === "number" && Math.abs(yaw - yawRef) > yawTol) return;

      const div = renderItem(p);
      div.onclick = () => {
        rgbImg.src = "/api/image?path=" + encodeURIComponent(p.rgb_path);
        tImg.src = p.t_path ? ("/api/image?path=" + encodeURIComponent(p.t_path)) : "";
        meta.textContent = JSON.stringify(p, null, 2);
      };
      list.appendChild(div);
    });
  }

  byId("onlyRTK").addEventListener("change", applyFilter);
  byId("yawTol").addEventListener("change", applyFilter);

  applyFilter();

  // 默认点开第一条
  if (poses.length){
    rgbImg.src = "/api/image?path=" + encodeURIComponent(poses[0].rgb_path);
    tImg.src = poses[0].t_path ? ("/api/image?path=" + encodeURIComponent(poses[0].t_path)) : "";
    meta.textContent = JSON.stringify(poses[0], null, 2);
  }
}

main();
