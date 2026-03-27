import init, {
    slope, aspect_degrees, hillshade_compute, multidirectional_hillshade,
    tpi_compute, tri_compute, fill_depressions, priority_flood_fill,
    flow_direction_d8, flow_accumulation_d8, twi_compute, hand_compute,
    morph_erode, morph_dilate, morph_opening, morph_closing,
    focal_mean, focal_std, focal_range,
    northness_compute, eastness_compute, dev_compute, geomorphons_compute,
    shape_index, curvedness, curvature_compute,
} from './wasm/surtgis_wasm.js';

import { searchCatalogs, fetchCollections, searchItems, extractAssets, downloadCOG, checkFileSize } from './stac.js';

// ── State ────────────────────────────────────────────────
let wasmReady = false;
let currentDem = null;
let currentResult = null;
let map = null;
let baseTileLayer = null;
let graticuleLayer = null;
let drawnBbox = null;       // [west, south, east, north]
let bboxRect = null;        // Leaflet rectangle on map
let drawControl = null;
let drawnItems = null;

let layerStack = [];
let layerIdCounter = 0;

// STAC state
let stacCatalogUrl = null;
let stacSelectedCollection = null;
let stacFoundItems = [];
let stacSelectedItem = null; // current STAC item (has bbox in metadata)

// ── DOM ──────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Base maps ────────────────────────────────────────────
const BASEMAPS = {
    osm:       'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    topo:      'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
    dark:      'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
};

// ── Init map ─────────────────────────────────────────────
map = L.map('map', { zoomControl: true }).setView([-33.5, -70.45], 5);
baseTileLayer = L.tileLayer(BASEMAPS.osm, { maxZoom: 19, attribution: '&copy; OSM' }).addTo(map);

// ── Leaflet Draw for bbox ────────────────────────────────
drawnItems = new L.FeatureGroup().addTo(map);
drawControl = new L.Control.Draw({
    draw: {
        polygon: false, polyline: false, circle: false, circlemarker: false, marker: false,
        rectangle: { shapeOptions: { color: '#f59e0b', weight: 2, fillOpacity: 0.1 } }
    },
    edit: { featureGroup: drawnItems }
});

map.on(L.Draw.Event.CREATED, (e) => {
    drawnItems.clearLayers();
    drawnItems.addLayer(e.layer);
    const b = e.layer.getBounds();
    drawnBbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()];
    $('bboxDisplay').textContent = drawnBbox.map(v => v.toFixed(4)).join(', ');
    updateStacSearchReady();
});

// ── Graticule ────────────────────────────────────────────
function createGraticule() {
    const group = L.layerGroup();
    const gs = { color: 'rgba(0,0,0,0.25)', weight: 0.8, dashArray: '4,4', fill: false };
    const ls = 'font:10px monospace;color:rgba(0,0,0,0.5);background:rgba(255,255,255,0.7);padding:0 2px;border-radius:2px;white-space:nowrap';
    function update() {
        group.clearLayers();
        const bounds = map.getBounds(); const zoom = map.getZoom();
        let iv; if(zoom>=14)iv=0.005;else if(zoom>=12)iv=0.01;else if(zoom>=10)iv=0.05;else if(zoom>=8)iv=0.1;else if(zoom>=6)iv=0.5;else if(zoom>=4)iv=2;else if(zoom>=2)iv=5;else iv=10;
        const d=iv>=1?0:iv>=0.1?1:iv>=0.01?2:3;
        const sLat=Math.floor(bounds.getSouth()/iv)*iv,eLat=Math.ceil(bounds.getNorth()/iv)*iv;
        const sLng=Math.floor(bounds.getWest()/iv)*iv,eLng=Math.ceil(bounds.getEast()/iv)*iv;
        for(let lat=sLat;lat<=eLat;lat+=iv){group.addLayer(L.polyline([[lat,sLng-1],[lat,eLng+1]],gs));group.addLayer(L.marker([lat,bounds.getWest()+(bounds.getEast()-bounds.getWest())*.02],{icon:L.divIcon({className:'',html:`<span style="${ls}">${lat.toFixed(d)}&deg;</span>`,iconSize:[0,0],iconAnchor:[0,6]})}))}
        for(let lng=sLng;lng<=eLng;lng+=iv){group.addLayer(L.polyline([[sLat-1,lng],[eLat+1,lng]],gs));group.addLayer(L.marker([bounds.getSouth()+(bounds.getNorth()-bounds.getSouth())*.02,lng],{icon:L.divIcon({className:'',html:`<span style="${ls}">${lng.toFixed(d)}&deg;</span>`,iconSize:[0,0],iconAnchor:[0,0]})}))}
    }
    map.on('moveend',update);map.on('zoomend',update);update();return group;
}
graticuleLayer = createGraticule().addTo(map);

// ── Mouse coords + pixel query ───────────────────────────
map.on('mousemove',(e)=>{
    const lat=e.latlng.lat,lon=e.latlng.lng;
    $('coords').textContent=`Lat: ${lat.toFixed(5)}, Lon: ${lon.toFixed(5)}`;
    let vt='';
    for(let i=layerStack.length-1;i>=0;i--){const l=layerStack[i];if(!l.visible||!l.rasterInfo)continue;const ri=l.rasterInfo;let sx=lon,sy=lat;if(ri.epsg&&ri.epsg!==4326){try{const sp=getProj4Def(ri.epsg);if(sp){const pt=proj4('EPSG:4326',sp,[lon,lat]);sx=pt[0];sy=pt[1]}}catch(_){continue}}const[mnX,mnY,mxX,mxY]=ri.bbox;if(sx<mnX||sx>mxX||sy<mnY||sy>mxY)continue;const px=Math.floor((sx-mnX)/(mxX-mnX)*ri.width),py=Math.floor((mxY-sy)/(mxY-mnY)*ri.height);if(px>=0&&px<ri.width&&py>=0&&py<ri.height){const v=ri.data[py*ri.width+px];vt=Number.isFinite(v)?`${l.name}: ${v.toFixed(4)}`:`${l.name}: NoData`}break}
    $('pixelValue').textContent=vt||'Value: --';
});

// ── Tabs ─────────────────────────────────────────────────
function switchTab(name) {
    ['Local','Stac','Layers'].forEach(t => {
        $('tab'+t).classList.toggle('text-blue-600', t===name);
        $('tab'+t).classList.toggle('border-b-2', t===name);
        $('tab'+t).classList.toggle('border-blue-600', t===name);
        $('tab'+t).classList.toggle('text-gray-500', t!==name);
        $('panel'+t).classList.toggle('hidden', t!==name);
    });
    if (name === 'Stac' && !map.hasLayer(drawControl)) {
        // Show draw hint
    }
}
$('tabLocal').addEventListener('click', ()=>switchTab('Local'));
$('tabStac').addEventListener('click', ()=>switchTab('Stac'));
$('tabLayers').addEventListener('click', ()=>switchTab('Layers'));

// ── Layer toggles ────────────────────────────────────────
$('layerGrid').addEventListener('change',(e)=>{e.target.checked?graticuleLayer.addTo(map):map.removeLayer(graticuleLayer)});
$('layerBase').addEventListener('change',(e)=>{e.target.checked?baseTileLayer.addTo(map):map.removeLayer(baseTileLayer)});
$('basemapSelect').addEventListener('change',(e)=>{const u=BASEMAPS[e.target.value];if(!u)return;map.removeLayer(baseTileLayer);baseTileLayer=L.tileLayer(u,{maxZoom:19,attribution:''}).addTo(map);if(!$('layerBase').checked)map.removeLayer(baseTileLayer)});
$('opacity').addEventListener('input',(e)=>{const v=parseInt(e.target.value);$('opacityVal').textContent=v+'%';layerStack.forEach(l=>{if(l.visible)l.overlay.setOpacity(v/100)})});

// ── Layer management ─────────────────────────────────────
function addRasterLayer(name,dataUrl,bounds,tiffBytes,rasterInfo){
    const id='layer_'+(layerIdCounter++);const overlay=L.imageOverlay(dataUrl,bounds,{opacity:parseInt($('opacity').value)/100}).addTo(map);
    const entry={id,name,overlay,bounds,visible:true,tiffBytes,rasterInfo:rasterInfo||null};
    layerStack.push(entry);syncZOrder();renderLayerPanel();map.fitBounds(bounds);return entry;
}
function syncZOrder(){for(let i=0;i<layerStack.length;i++){const l=layerStack[i];if(l.visible&&l.overlay._image)l.overlay._image.style.zIndex=200+i}}
function renderLayerPanel(){
    const c=$('rasterLayers');c.innerHTML='';
    for(let i=layerStack.length-1;i>=0;i--){const l=layerStack[i];const isT=i===layerStack.length-1,isB=i===0;const r=document.createElement('div');r.className='flex items-center gap-0.5 py-0.5 group';
    r.innerHTML=`<input type="checkbox" data-lid="${l.id}" ${l.visible?'checked':''}><span class="flex-1 truncate text-gray-700 text-xs" title="${l.name}">${l.name}</span><button data-up="${l.id}" class="text-gray-400 hover:text-blue-600 text-xs px-0.5 ${isT?'invisible':''}" title="Up">&#9650;</button><button data-dn="${l.id}" class="text-gray-400 hover:text-blue-600 text-xs px-0.5 ${isB?'invisible':''}" title="Down">&#9660;</button><button data-rm="${l.id}" class="text-red-400 hover:text-red-600 opacity-0 group-hover:opacity-100 text-xs px-0.5">&times;</button>`;
    c.appendChild(r)}
    c.querySelectorAll('input[data-lid]').forEach(cb=>{cb.addEventListener('change',e=>{const l=layerStack.find(x=>x.id===e.target.dataset.lid);if(!l)return;l.visible=e.target.checked;l.visible?l.overlay.addTo(map):map.removeLayer(l.overlay);syncZOrder()})});
    c.querySelectorAll('button[data-rm]').forEach(b=>{b.addEventListener('click',e=>{const idx=layerStack.findIndex(x=>x.id===e.target.dataset.rm);if(idx===-1)return;map.removeLayer(layerStack[idx].overlay);layerStack.splice(idx,1);renderLayerPanel()})});
    c.querySelectorAll('button[data-up]').forEach(b=>{b.addEventListener('click',e=>{const idx=layerStack.findIndex(x=>x.id===e.target.dataset.up);if(idx<0||idx>=layerStack.length-1)return;[layerStack[idx],layerStack[idx+1]]=[layerStack[idx+1],layerStack[idx]];syncZOrder();renderLayerPanel()})});
    c.querySelectorAll('button[data-dn]').forEach(b=>{b.addEventListener('click',e=>{const idx=layerStack.findIndex(x=>x.id===e.target.dataset.dn);if(idx<=0)return;[layerStack[idx],layerStack[idx-1]]=[layerStack[idx-1],layerStack[idx]];syncZOrder();renderLayerPanel()})});
}
$('clearLayersBtn').addEventListener('click',()=>{layerStack.forEach(l=>map.removeLayer(l.overlay));layerStack=[];renderLayerPanel();let o=$('legend');if(o)o.remove()});

// ── Init WASM ────────────────────────────────────────────
(async()=>{try{await init();wasmReady=true;setStatus('WASM loaded. Ready.')}catch(e){setStatus('WASM init failed: '+e.message)}})();
function setStatus(m){$('status').textContent=m}
function enableCompute(){$('computeBtn').disabled=!(wasmReady&&currentDem)}

// ── CRS ──────────────────────────────────────────────────
function detectEPSG(image){const gk=image.getGeoKeys?image.getGeoKeys():{};if(gk.ProjectedCSTypeGeoKey&&gk.ProjectedCSTypeGeoKey!==32767)return gk.ProjectedCSTypeGeoKey;if(gk.GeographicTypeGeoKey&&gk.GeographicTypeGeoKey!==32767)return gk.GeographicTypeGeoKey;const bb=image.getBoundingBox();if(bb&&Math.abs(bb[0])<=360&&Math.abs(bb[2])<=360&&Math.abs(bb[1])<=90&&Math.abs(bb[3])<=90)return 4326;return null}
function bboxToLeafletBounds(bbox,epsg){const[mnX,mnY,mxX,mxY]=bbox;if(!epsg||epsg===4326)return[[mnY,mnX],[mxY,mxX]];try{const s=getProj4Def(epsg);if(s){const sw=proj4(s,'EPSG:4326',[mnX,mnY]),ne=proj4(s,'EPSG:4326',[mxX,mxY]);return[[sw[1],sw[0]],[ne[1],ne[0]]]}}catch(e){console.warn('Reproj:',e)}if(Math.abs(mnX)>360||Math.abs(mxY)>360)return null;return[[mnY,mnX],[mxY,mxX]]}
function getProj4Def(e){if(e>=32601&&e<=32660)return`+proj=utm +zone=${e-32600} +datum=WGS84 +units=m +no_defs`;if(e>=32701&&e<=32760)return`+proj=utm +zone=${e-32700} +south +datum=WGS84 +units=m +no_defs`;if(e===3857)return'+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +no_defs';if(e===4269)return'+proj=longlat +datum=NAD83 +no_defs';return null}

// ── Load Demo DEM ────────────────────────────────────────
$('loadDemoBtn').addEventListener('click',async()=>{setStatus('Loading demo DEM...');try{const r=await fetch('./data/demo_dem.tif');const buf=await r.arrayBuffer();currentDem=new Uint8Array(buf);await showOnMap(buf,'DEM (demo)','terrain');$('fileStatus').textContent=`Demo (${(buf.byteLength/1024).toFixed(0)} KB)`;setStatus('Demo loaded.');enableCompute()}catch(e){setStatus('Failed: '+e.message)}});

// ── Upload file ──────────────────────────────────────────
$('demFile').addEventListener('change',async(e)=>{const f=e.target.files[0];if(!f)return;setStatus('Loading '+f.name+'...');try{const buf=await f.arrayBuffer();currentDem=new Uint8Array(buf);await showOnMap(buf,f.name,'terrain');$('fileStatus').textContent=`${f.name} (${(buf.byteLength/1024/1024).toFixed(1)} MB)`;setStatus('Loaded.');enableCompute()}catch(e2){setStatus('Error: '+e2.message)}});

// ── Show buffer on map ───────────────────────────────────
async function showOnMap(buffer, name, cmapName, fallbackBbox) {
    const tiff=await GeoTIFF.fromArrayBuffer(buffer);const image=await tiff.getImage();
    const w=image.getWidth(),h=image.getHeight();
    const data=(await image.readRasters())[0];

    // Try native bbox from GeoTIFF affine transform
    let bbox=null, epsg=null;
    try {
        bbox=image.getBoundingBox();
        epsg=detectEPSG(image);
    } catch(_) {
        // No affine transform (e.g. Sentinel-1 GRD with GCPs)
        console.warn('No affine transform, using fallback bbox');
    }

    // Fallback: use STAC item bbox (always WGS84)
    if (!bbox && fallbackBbox) {
        bbox = fallbackBbox;
        epsg = 4326;
        setStatus(`Loaded (using STAC bbox, no native geotransform)`);
    }

    if (!bbox) {
        setStatus('No geolocation available. Compute will still work.');
        // Still set as current DEM for algorithms
        return;
    }

    const bounds=bboxToLeafletBounds(bbox,epsg);
    if(bounds){
        const du=rasterToDataURL(data,w,h,cmapName);
        addRasterLayer(name,du,bounds,new Uint8Array(buffer),{data,width:w,height:h,bbox,epsg});
    } else {
        setStatus('Unknown CRS, overlay skipped.');
    }
}

// ── Compute ──────────────────────────────────────────────
$('computeBtn').addEventListener('click',()=>{
    if(!wasmReady||!currentDem)return;const algo=$('algorithm').value,rad=parseInt($('radius').value)||3,az=parseFloat($('azimuth').value)||315;
    $('computeBtn').disabled=true;setStatus('Computing '+algo+'...');
    setTimeout(async()=>{try{const t0=performance.now();let r;
    switch(algo){
        case 'slope':r=slope(currentDem,'degrees');break;case 'aspect':r=aspect_degrees(currentDem);break;
        case 'hillshade':r=hillshade_compute(currentDem,az,45);break;case 'multidirectional_hillshade':r=multidirectional_hillshade(currentDem);break;
        case 'tpi':r=tpi_compute(currentDem,rad);break;case 'tri':r=tri_compute(currentDem);break;
        case 'northness':r=northness_compute(currentDem);break;case 'eastness':r=eastness_compute(currentDem);break;
        case 'curvature_general':r=curvature_compute(currentDem,'general');break;case 'curvature_profile':r=curvature_compute(currentDem,'profile');break;case 'curvature_plan':r=curvature_compute(currentDem,'plan');break;
        case 'dev':r=dev_compute(currentDem,rad);break;case 'geomorphons':r=geomorphons_compute(currentDem,1.0,rad);break;
        case 'shape_index':r=shape_index(currentDem);break;case 'curvedness':r=curvedness(currentDem);break;
        case 'fill':r=fill_depressions(currentDem);break;case 'priority_flood':r=priority_flood_fill(currentDem);break;
        case 'flow_direction':r=flow_direction_d8(currentDem);break;case 'flow_accumulation':{const fd=flow_direction_d8(currentDem);r=flow_accumulation_d8(fd);break}
        case 'twi':r=twi_compute(currentDem);break;case 'hand':r=hand_compute(currentDem,1000);break;
        case 'erode':r=morph_erode(currentDem,rad);break;case 'dilate':r=morph_dilate(currentDem,rad);break;
        case 'opening':r=morph_opening(currentDem,rad);break;case 'closing':r=morph_closing(currentDem,rad);break;
        case 'focal_mean':r=focal_mean(currentDem,rad);break;case 'focal_std':r=focal_std(currentDem,rad);break;case 'focal_range':r=focal_range(currentDem,rad);break;
        default:throw new Error('Unknown: '+algo)}
    const el=((performance.now()-t0)/1000).toFixed(2);currentResult=r;
    const cm=$('colormap').value==='auto'?autoColormap(algo):$('colormap').value;
    await showResultOnMap(r,algo,cm);setStatus(algo+' done in '+el+'s');$('downloadBtn').disabled=false;
    }catch(e){setStatus('Error: '+e.message);console.error(e)}finally{enableCompute()}},50);
});

async function showResultOnMap(tiffBytes, name, cmapName) {
    const tiff=await GeoTIFF.fromArrayBuffer(tiffBytes.buffer);const image=await tiff.getImage();
    const w=image.getWidth(),h=image.getHeight(),bbox=image.getBoundingBox();
    const data=(await image.readRasters())[0];const epsg=detectEPSG(image);const bounds=bboxToLeafletBounds(bbox,epsg);
    if(bounds){const du=rasterToDataURL(data,w,h,cmapName);addRasterLayer(name,du,bounds,tiffBytes,{data,width:w,height:h,bbox,epsg})}
    showLegend(name,data,cmapName);
}

$('downloadBtn').addEventListener('click',()=>{if(!currentResult)return;const b=new Blob([currentResult],{type:'application/octet-stream'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='surtgis_'+$('algorithm').value+'.tif';a.click();URL.revokeObjectURL(a.href)});

// ════════════════════════════════════════════════════════
// STAC PANEL
// ════════════════════════════════════════════════════════

// Quick catalog buttons
document.querySelectorAll('.stac-quick').forEach(btn => {
    btn.addEventListener('click', async () => {
        stacCatalogUrl = btn.dataset.stacUrl;
        $('stacCatalogStatus').textContent = 'Catalog: ' + btn.textContent.trim();

        if (btn.dataset.collection) {
            // Quick shortcut: pre-select collection
            stacSelectedCollection = btn.dataset.collection;
            $('stacCollection').innerHTML = `<option value="${btn.dataset.collection}">${btn.dataset.collection}</option>`;
            $('stacCollection').disabled = false;
            $('stacCollectionInfo').textContent = 'Quick: ' + btn.dataset.collection;
            updateStacSearchReady();
        } else {
            // Load collections
            await loadCollections(stacCatalogUrl);
        }
    });
});

// Search catalogs
$('stacSearchBtn').addEventListener('click', async () => {
    const kw = $('stacSearch').value.trim();
    if (!kw) return;
    $('stacCatalogStatus').textContent = 'Searching...';
    try {
        const cats = await searchCatalogs(kw);
        const container = $('stacCatalogResults');
        container.classList.remove('hidden');
        if (cats.length === 0) {
            container.innerHTML = '<p class="p-2 text-gray-400">No catalogs found</p>';
        } else {
            container.innerHTML = cats.map(c =>
                `<div class="p-1.5 hover:bg-blue-50 cursor-pointer border-b border-gray-100" data-cat-url="${c.url}">
                    <div class="font-medium">${c.title}</div>
                    <div class="text-gray-400 truncate">${c.description}</div>
                    <div class="text-blue-500 truncate">${c.url}</div>
                </div>`
            ).join('');
            container.querySelectorAll('[data-cat-url]').forEach(el => {
                el.addEventListener('click', async () => {
                    stacCatalogUrl = el.dataset.catUrl;
                    $('stacCatalogStatus').textContent = 'Catalog: ' + el.querySelector('.font-medium').textContent;
                    container.classList.add('hidden');
                    await loadCollections(stacCatalogUrl);
                });
            });
        }
        $('stacCatalogStatus').textContent = `Found ${cats.length} catalogs`;
    } catch (e) {
        $('stacCatalogStatus').textContent = 'Error: ' + e.message;
    }
});

// Load collections
async function loadCollections(catalogUrl) {
    $('stacCollection').disabled = true;
    $('stacCollectionInfo').textContent = 'Loading collections...';
    try {
        const cols = await fetchCollections(catalogUrl);
        const sel = $('stacCollection');
        sel.innerHTML = '<option value="">-- select --</option>' + cols.map(c =>
            `<option value="${c.id}" title="${c.description}">${c.title}</option>`
        ).join('');
        sel.disabled = false;
        $('stacCollectionInfo').textContent = `${cols.length} collections`;
    } catch (e) {
        $('stacCollectionInfo').textContent = 'Error: ' + e.message;
    }
}

$('stacCollection').addEventListener('change', () => {
    stacSelectedCollection = $('stacCollection').value;
    updateStacSearchReady();
});

// Draw bbox button
$('drawBboxBtn').addEventListener('click', () => {
    // Enable rectangle draw mode
    new L.Draw.Rectangle(map, drawControl.options.draw.rectangle).enable();
    setStatus('Draw a rectangle on the map for search area');
});

function updateStacSearchReady() {
    $('stacSearchItems').disabled = !(stacCatalogUrl && stacSelectedCollection && drawnBbox);
}

// Search items
$('stacSearchItems').addEventListener('click', async () => {
    if (!stacCatalogUrl || !stacSelectedCollection || !drawnBbox) return;
    const dateFrom = $('stacDateFrom').value;
    const dateTo = $('stacDateTo').value;
    const datetime = dateFrom + '/' + dateTo;
    const maxItems = parseInt($('stacMaxItems').value) || 5;

    $('stacItemsStatus').textContent = 'Searching...';
    setStatus('Searching STAC items...');

    try {
        stacFoundItems = await searchItems(stacCatalogUrl, stacSelectedCollection, drawnBbox, datetime, maxItems);
        const container = $('stacResults');

        if (stacFoundItems.length === 0) {
            container.innerHTML = '<p class="text-gray-400 p-2">No items found. Try wider bbox or date range.</p>';
            $('stacItemsStatus').textContent = '0 items found';
        } else {
            container.innerHTML = stacFoundItems.map((item, idx) =>
                `<div class="p-1.5 hover:bg-blue-50 cursor-pointer border-b border-gray-100 stac-item" data-idx="${idx}">
                    <div class="font-medium text-xs">${item.id}</div>
                    <div class="text-gray-400 text-xs">${item.properties?.datetime || 'no date'}</div>
                    <div class="text-gray-400 text-xs">${Object.keys(item.assets || {}).length} assets</div>
                </div>`
            ).join('');

            // Click item → show assets
            container.querySelectorAll('.stac-item').forEach(el => {
                el.addEventListener('click', () => {
                    const idx = parseInt(el.dataset.idx);
                    selectStacItem(stacFoundItems[idx]);
                    // Highlight
                    container.querySelectorAll('.stac-item').forEach(x => x.classList.remove('bg-blue-100'));
                    el.classList.add('bg-blue-100');
                });
            });

            // Auto-select first item
            selectStacItem(stacFoundItems[0]);
            container.querySelector('.stac-item').classList.add('bg-blue-100');

            $('stacItemsStatus').textContent = `${stacFoundItems.length} items found`;
        }
        setStatus(`Found ${stacFoundItems.length} STAC items`);
    } catch (e) {
        $('stacItemsStatus').textContent = 'Error: ' + e.message;
        setStatus('STAC search error: ' + e.message);
        console.error(e);
    }
});

function selectStacItem(item) {
    stacSelectedItem = item;
    const assets = extractAssets(item);
    const sel = $('stacAsset');
    sel.innerHTML = assets.map(a =>
        `<option value="${a.href}" title="${a.type}">${a.key} - ${a.title}</option>`
    ).join('');
    sel.disabled = assets.length === 0;
    $('stacDownloadBtn').disabled = assets.length === 0;
}

// Download COG
const MAX_DOWNLOAD_MB = 100;

$('stacDownloadBtn').addEventListener('click', async () => {
    const href = $('stacAsset').value;
    if (!href) return;

    $('stacDownloadBtn').disabled = true;
    $('stacDownloadStatus').textContent = 'Checking file size...';
    setStatus('Checking asset size...');

    try {
        // Check size first
        const fileSize = await checkFileSize(href);
        if (fileSize) {
            const mb = fileSize / 1024 / 1024;
            if (mb > MAX_DOWNLOAD_MB) {
                $('stacDownloadStatus').textContent = `Too large: ${mb.toFixed(0)} MB (max ${MAX_DOWNLOAD_MB} MB). Use CLI for large files.`;
                setStatus(`File is ${mb.toFixed(0)} MB — too large for browser. Use surtgis CLI instead.`);
                $('stacDownloadBtn').disabled = false;
                return;
            }
            $('stacDownloadStatus').textContent = `Downloading ${mb.toFixed(1)} MB...`;
        } else {
            $('stacDownloadStatus').textContent = 'Downloading...';
        }

        setStatus('Downloading COG asset...');
        const buffer = await downloadCOG(href);
        const mbDown = (buffer.byteLength / 1024 / 1024).toFixed(1);
        $('stacDownloadStatus').textContent = `Downloaded (${mbDown} MB)`;

        // Load onto map — use STAC item bbox as fallback for files without affine
        const assetName = $('stacAsset').selectedOptions[0]?.textContent || 'STAC asset';
        const itemBbox = stacSelectedItem?.bbox || null;
        await showOnMap(buffer, assetName, 'terrain', itemBbox);

        // Set as current DEM for algorithms AND for download button
        currentDem = new Uint8Array(buffer);
        currentResult = new Uint8Array(buffer);
        enableCompute();
        $('downloadBtn').disabled = false;

        setStatus('STAC asset loaded! Switch to Local tab to run algorithms, or Download.');
    } catch (e) {
        $('stacDownloadStatus').textContent = 'Error: ' + e.message;
        setStatus('Download error: ' + e.message);
        console.error(e);
    } finally {
        $('stacDownloadBtn').disabled = false;
    }
});

// ════════════════════════════════════════════════════════
// RENDERING
// ════════════════════════════════════════════════════════

function rasterToDataURL(data,w,h,cmapName){const canvas=document.createElement('canvas');canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d');const img=ctx.createImageData(w,h);const px=img.data;let lo=Infinity,hi=-Infinity;for(let i=0;i<data.length;i++){const v=data[i];if(Number.isFinite(v)){if(v<lo)lo=v;if(v>hi)hi=v}}if(hi===lo)hi=lo+1;const range=hi-lo;const colors=COLORMAPS[cmapName]||COLORMAPS.viridis;for(let i=0;i<data.length;i++){const v=data[i];let t=0,alpha=255;if(Number.isFinite(v)){t=(v-lo)/range;t=t<0?0:t>1?1:t}else{alpha=0}const idx=Math.min(Math.floor(t*255),255);const c=colors[idx];px[i*4]=c[0];px[i*4+1]=c[1];px[i*4+2]=c[2];px[i*4+3]=alpha}ctx.putImageData(img,0,0);return canvas.toDataURL()}

function showLegend(name,data,cmapName){let o=$('legend');if(o)o.remove();let lo=Infinity,hi=-Infinity;for(let i=0;i<data.length;i++){const v=data[i];if(Number.isFinite(v)){if(v<lo)lo=v;if(v>hi)hi=v}}const bar=document.createElement('canvas');bar.width=200;bar.height=12;const bctx=bar.getContext('2d');const colors=COLORMAPS[cmapName]||COLORMAPS.viridis;for(let x=0;x<200;x++){const idx=Math.floor((x/199)*255);const c=colors[idx];bctx.fillStyle=`rgb(${c[0]},${c[1]},${c[2]})`;bctx.fillRect(x,0,1,12)}const div=document.createElement('div');div.id='legend';div.style.cssText='position:absolute;bottom:24px;left:50px;background:rgba(255,255,255,0.92);padding:8px 12px;border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.2);font:11px system-ui;z-index:1000;';div.innerHTML=`<div style="font-weight:600;margin-bottom:3px">${name}</div>`;div.appendChild(bar);div.innerHTML+=`<div style="display:flex;justify-content:space-between;font-size:10px;margin-top:2px"><span>${lo.toFixed(1)}</span><span>${hi.toFixed(1)}</span></div>`;$('map').appendChild(div)}

function autoColormap(a){return({slope:'viridis',aspect:'hsv',hillshade:'gray',multidirectional_hillshade:'gray',tpi:'seismic',tri:'viridis',northness:'seismic',eastness:'seismic',curvature_general:'seismic',curvature_profile:'seismic',curvature_plan:'seismic',dev:'seismic',geomorphons:'hsv',shape_index:'viridis',curvedness:'viridis',fill:'terrain',priority_flood:'terrain',flow_direction:'hsv',flow_accumulation:'plasma',twi:'viridis',hand:'viridis',erode:'gray',dilate:'gray',opening:'gray',closing:'gray',focal_mean:'viridis',focal_std:'plasma',focal_range:'plasma'})[a]||'viridis'}

const COLORMAPS={viridis:buildViridis(),plasma:buildPlasma(),terrain:buildTerrain(),gray:buildGray(),seismic:buildSeismic(),hsv:buildHSV()};
function buildViridis(){return interp([[0,68,1,84],[.25,59,82,139],[.5,33,145,140],[.75,94,201,98],[1,253,231,37]])}
function buildPlasma(){return interp([[0,13,8,135],[.25,126,3,168],[.5,204,71,81],[.75,249,149,64],[1,240,249,33]])}
function buildTerrain(){return interp([[0,51,102,0],[.2,102,153,0],[.4,204,204,102],[.6,153,102,51],[.8,128,128,128],[1,255,255,255]])}
function buildGray(){const c=[];for(let i=0;i<256;i++)c.push([i,i,i]);return c}
function buildSeismic(){return interp([[0,0,0,153],[.25,77,133,222],[.5,255,255,255],[.75,222,77,77],[1,153,0,0]])}
function buildHSV(){const c=[];for(let i=0;i<256;i++){const h=i/255;const[r,g,b]=hsl2rgb(h,1,.5);c.push([r,g,b])}return c}
function interp(stops){const c=[];for(let i=0;i<256;i++){const t=i/255;let s0=stops[0],s1=stops[stops.length-1];for(let j=0;j<stops.length-1;j++){if(t>=stops[j][0]&&t<=stops[j+1][0]){s0=stops[j];s1=stops[j+1];break}}const f=s1[0]===s0[0]?0:(t-s0[0])/(s1[0]-s0[0]);c.push([Math.round(s0[1]+f*(s1[1]-s0[1])),Math.round(s0[2]+f*(s1[2]-s0[2])),Math.round(s0[3]+f*(s1[3]-s0[3]))])}return c}
function hsl2rgb(h,s,l){let r,g,b;if(s===0){r=g=b=l}else{const hue2rgb=(p,q,t)=>{if(t<0)t+=1;if(t>1)t-=1;if(t<1/6)return p+(q-p)*6*t;if(t<1/2)return q;if(t<2/3)return p+(q-p)*(2/3-t)*6;return p};const q=l<.5?l*(1+s):l+s-l*s;const p=2*l-q;r=hue2rgb(p,q,h+1/3);g=hue2rgb(p,q,h);b=hue2rgb(p,q,h-1/3)}return[Math.round(r*255),Math.round(g*255),Math.round(b*255)]}
