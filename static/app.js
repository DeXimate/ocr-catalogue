const state={job:null,products:[],timer:null,dirty:false};
const $=selector=>document.querySelector(selector);

async function api(url,options={}){
  const response=await fetch(url,{headers:{'Content-Type':'application/json',...(options.headers||{})},...options});
  if(!response.ok)throw new Error((await response.json().catch(()=>({}))).error||response.statusText);
  return response.json();
}

function esc(value){return String(value??'').replace(/[&<>"']/g,char=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]))}
function asset(path){return path?`/api/assets/${state.job.id}/${encodeURIComponent(path).replaceAll('%2F','/')}`:''}
function selected(){return state.products.filter(product=>product.selected)}

function toast(message){
  const element=$('#toast');
  element.textContent=message;
  element.classList.add('show');
  clearTimeout(element.timer);
  element.timer=setTimeout(()=>element.classList.remove('show'),2400);
}

function setDirty(dirty){
  state.dirty=dirty;
  const indicator=$('#saveState');
  indicator.textContent=dirty?'Modifications non enregistrées':'Toutes les modifications sont enregistrées';
  indicator.classList.toggle('dirty',dirty);
}

function closeOnBackdrop(dialog){
  dialog.addEventListener('click',event=>{
    const bounds=dialog.getBoundingClientRect();
    const outside=event.clientX<bounds.left||event.clientX>bounds.right||event.clientY<bounds.top||event.clientY>bounds.bottom;
    if(outside)dialog.close();
  });
}

async function loadJobs(){
  const jobs=await api('/api/jobs');
  $('#jobs').innerHTML='<option value="">Choisir un traitement</option>'+jobs.map(job=>`<option value="${job.id}">${esc(job.filename)} — ${esc(job.status)}</option>`).join('');
  const remembered=localStorage.getItem('ocr_catalogue_selected_job');
  const initial=jobs.find(job=>job.id===remembered)||jobs.find(job=>job.status==='Terminé');
  if(initial){
    $('#jobs').value=initial.id;
    localStorage.setItem('ocr_catalogue_selected_job',initial.id);
    await loadJob(initial.id);
  }
}

async function loadJob(id){
  if(!id){state.job=null;state.products=[];render();return}
  state.job=await api(`/api/jobs/${id}`);
  state.products=(state.job.products||[]).map(product=>({...product,pourcentage:product.pourcentage||product.remise||''}));
  setDirty(false);
  render();
  if(['Importé','Traitement'].includes(state.job.status)){
    showProgress();
    clearTimeout(state.timer);
    state.timer=setTimeout(()=>loadJob(id),1200);
  }
}

function showProgress(){
  const element=$('#progress');
  if(!state.job||state.job.status==='Terminé'){element.classList.add('hidden');return}
  element.classList.remove('hidden');
  element.querySelector('span').style.width=`${state.job.progress||3}%`;
  element.querySelector('p').textContent=`${state.job.status} — ${state.job.progress||0}%`;
}

function visible(){
  const query=$('#search').value.trim().toLowerCase();
  const status=$('#status').value;
  return state.products.filter(product=>(!query||`${product.produit} ${product.marque}`.toLowerCase().includes(query))&&(!status||product.statut===status));
}

function editableCell(product,key){return `<td data-column="${key}"><input data-key="${key}" aria-label="${key}" value="${esc(product[key])}"></td>`}

function render(){
  showProgress();
  const rows=visible();
  const hasJob=Boolean(state.job);
  const selection=selected().length;
  $('#empty').style.display=rows.length?'none':'block';
  $('#rows').innerHTML=rows.map(product=>`
    <tr data-id="${product.id}" class="${product.statut==='Validé'?'':'review-row'}">
      <td class="check-cell"><input type="checkbox" data-key="selected" aria-label="Sélectionner ${esc(product.produit)}" ${product.selected?'checked':''}></td>
      <td><img class="thumb" src="${asset(product.photo)}" data-source="${asset(product.source_crop||product.photo)}" alt="${esc(product.produit)}" loading="lazy"></td>
      ${['produit','marque','quantite','prix_promo','pourcentage','promotion'].map(key=>editableCell(product,key)).join('')}
      <td>${product.page}</td>
      <td><span class="confidence ${product.confiance>=85?'valid':''}">${product.confiance}%</span></td>
      <td><select class="status-select" data-key="statut" aria-label="Statut de ${esc(product.produit)}"><option ${product.statut==='À vérifier'?'selected':''}>À vérifier</option><option ${product.statut==='Validé'?'selected':''}>Validé</option></select></td>
    </tr>`).join('');

  const review=state.products.filter(product=>product.statut!=='Validé').length;
  const average=state.products.length?Math.round(state.products.reduce((sum,product)=>sum+product.confiance,0)/state.products.length):0;
  $('#count').textContent=state.products.length;
  $('#review').textContent=review;
  $('#average').textContent=`${average}%`;
  $('#visibleCount').textContent=`${rows.length} produit${rows.length>1?'s':''}`;
  $('#filterHint').textContent=rows.length!==state.products.length?' selon les filtres':' dans ce catalogue';
  $('#selectedCount').textContent=selection;
  $('#selectedCount').classList.toggle('hidden',!selection);
  $('#validate').disabled=!selection;
  $('#save').disabled=!hasJob;
  $('#export').disabled=!hasJob;
  $('#all').checked=Boolean(rows.length)&&rows.every(product=>product.selected);
  $('#all').indeterminate=rows.some(product=>product.selected)&&!rows.every(product=>product.selected);
}

$('#rows').addEventListener('input',event=>{
  const row=event.target.closest('tr');
  const product=state.products.find(item=>item.id===row?.dataset.id);
  const key=event.target.dataset.key;
  if(!product||!key)return;
  product[key]=event.target.type==='checkbox'?event.target.checked:event.target.value;
  if(key!=='selected')setDirty(true);
  if(key==='statut'||key==='selected')render();
});

$('#rows').addEventListener('click',event=>{
  if(!event.target.classList.contains('thumb'))return;
  $('#preview img').src=event.target.dataset.source;
  $('#preview h3').textContent=event.target.alt;
  $('#preview').showModal();
});

$('#preview .close').onclick=()=>$('#preview').close();
closeOnBackdrop($('#preview'));
closeOnBackdrop($('#exportDialog'));
$('#file').onchange=async event=>{
  const file=event.target.files[0];
  if(!file)return;
  const data=new FormData();data.append('file',file);
  try{
    const response=await fetch('/api/import',{method:'POST',body:data});
    const job=await response.json();
    if(!response.ok)throw new Error(job.error);
    await loadJobs();
    $('#jobs').value=job.id;
    localStorage.setItem('ocr_catalogue_selected_job',job.id);
    await loadJob(job.id);
    toast('Catalogue importé — extraction en cours');
  }catch(error){toast(`Import impossible : ${error.message}`)}
  event.target.value='';
};

$('#jobs').onchange=event=>{
  if(event.target.value)localStorage.setItem('ocr_catalogue_selected_job',event.target.value);
  else localStorage.removeItem('ocr_catalogue_selected_job');
  loadJob(event.target.value);
};
$('#search').oninput=render;
$('#status').onchange=render;
$('#all').onchange=event=>{visible().forEach(product=>product.selected=event.target.checked);render()};
$('#validate').onclick=()=>{selected().forEach(product=>product.statut='Validé');setDirty(true);render();toast('Sélection validée')};
$('#save').onclick=async()=>{
  if(!state.job)return;
  try{await api(`/api/jobs/${state.job.id}/products`,{method:'PUT',body:JSON.stringify({products:state.products})});setDirty(false);toast('Modifications enregistrées')}
  catch(error){toast(`Enregistrement impossible : ${error.message}`)}
};
$('#export').onclick=()=>{if(state.job)$('#exportDialog').showModal()};
$('#confirmExport').onclick=async event=>{
  event.preventDefault();
  const button=event.currentTarget;
  button.disabled=true;button.textContent='Création…';
  try{
    await api(`/api/jobs/${state.job.id}/products`,{method:'PUT',body:JSON.stringify({products:state.products})});
    const data=await api(`/api/jobs/${state.job.id}/export`,{method:'POST',body:JSON.stringify({format:$('#format').value,include_photos:$('#photos').checked,scope:$('#scope').value})});
    setDirty(false);$('#exportDialog').close();
    const link=document.createElement('a');link.href=data.url;link.download=data.filename;link.click();
    toast('Export prêt au téléchargement');
  }catch(error){toast(`Export impossible : ${error.message}`)}
  finally{button.disabled=false;button.textContent='Créer l’export'}
};

loadJobs().then(render).catch(error=>toast(`Chargement impossible : ${error.message}`));
