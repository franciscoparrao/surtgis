<script>
  /** @type {{ onload: (payload: { name: string, bytes: Uint8Array }) => void, label?: string, accept?: string }} */
  let { onload, label = "Drop a GeoTIFF here or click to browse", accept = ".tif,.tiff" } = $props();

  let dragging = $state(false);
  let fileName = $state("");
  let fileSize = $state("");

  /** @type {HTMLInputElement} */
  let inputEl;

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1048576).toFixed(1)} MB`;
  }

  async function handleFile(file) {
    if (!file) return;
    fileName = file.name;
    fileSize = formatSize(file.size);
    const buf = await file.arrayBuffer();
    onload({ name: file.name, bytes: new Uint8Array(buf) });
  }

  function onDrop(e) {
    e.preventDefault();
    dragging = false;
    const file = e.dataTransfer?.files[0];
    handleFile(file);
  }

  function onDragOver(e) {
    e.preventDefault();
    dragging = true;
  }

  function onDragLeave() {
    dragging = false;
  }

  function onChange(e) {
    const file = e.target.files?.[0];
    handleFile(file);
  }
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<!-- svelte-ignore a11y_click_events_have_key_events -->
<div
  class="dropzone"
  class:dragging
  class:loaded={!!fileName}
  ondrop={onDrop}
  ondragover={onDragOver}
  ondragleave={onDragLeave}
  onclick={() => inputEl.click()}
>
  <input
    bind:this={inputEl}
    type="file"
    {accept}
    onchange={onChange}
    hidden
  />

  {#if fileName}
    <span class="icon">&#x2705;</span>
    <span class="name">{fileName}</span>
    <span class="size">{fileSize}</span>
  {:else}
    <span class="icon">&#x1F4C2;</span>
    <span class="label">{label}</span>
  {/if}
</div>

<style>
  .dropzone {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.35rem;
    padding: 1.2rem 1rem;
    border: 2px dashed var(--border);
    border-radius: 10px;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    text-align: center;
  }

  .dropzone:hover,
  .dropzone.dragging {
    border-color: var(--accent);
    background: rgba(108, 138, 255, 0.06);
  }

  .dropzone.loaded {
    border-style: solid;
    border-color: var(--success);
  }

  .icon {
    font-size: 1.5rem;
  }

  .label {
    color: var(--text-muted);
    font-size: 0.82rem;
  }

  .name {
    font-weight: 600;
    font-size: 0.85rem;
    word-break: break-all;
  }

  .size {
    color: var(--text-muted);
    font-size: 0.75rem;
  }
</style>
