# Build script for Tiny Llama WASM
# Run from the project root: .\build_wasm.ps1

$EMSDK = "D:\Files\emsdk"
$EMCC = "$EMSDK\upstream\emscripten\emcc.bat"

Write-Host "=== Building Tiny Llama WASM ===" -ForegroundColor Cyan

# Step 1: Compile C -> WASM
Write-Host "`n[1/3] Compiling run_wasm.c -> web/llama.js + llama.wasm..." -ForegroundColor Yellow
& $EMCC -O3 run_wasm.c -o web/llama.js -lm `
    -s EXPORTED_FUNCTIONS="['_init_model','_generate_next','_get_logits_ptr','_get_vocab_size','_encode_char','_decode_token','_reset_state','_cleanup','_malloc','_free']" `
    -s EXPORTED_RUNTIME_METHODS="['ccall','cwrap','HEAPU8','HEAPF32']" `
    -s ALLOW_MEMORY_GROWTH=1 `
    -s INITIAL_MEMORY=67108864 `
    -s MODULARIZE=1 `
    -s EXPORT_NAME="Module" `
    -s ENVIRONMENT=web

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: WASM compilation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  -> web/llama.js + web/llama.wasm created" -ForegroundColor Green

# Step 2: Copy model files to web/
Write-Host "`n[2/3] Copying model files to web/..." -ForegroundColor Yellow
Copy-Item "out\tiny_llama.bin" "web\tiny_llama.bin" -Force
Copy-Item "out\tokenizer.bin" "web\tokenizer.bin" -Force
Write-Host "  -> web/tiny_llama.bin + web/tokenizer.bin copied" -ForegroundColor Green

# Step 3: Print summary
Write-Host "`n[3/3] Build complete!" -ForegroundColor Cyan
$wasmSize = (Get-Item "web\llama.wasm").Length / 1024
$modelSize = (Get-Item "web\tiny_llama.bin").Length / 1024 / 1024
Write-Host "  WASM size: $([math]::Round($wasmSize, 1)) KB"
Write-Host "  Model size: $([math]::Round($modelSize, 2)) MB"
Write-Host "`nTo run locally:" -ForegroundColor Yellow
Write-Host "  python -m http.server 8080 --directory web"
Write-Host "  Then open: http://localhost:8080"
