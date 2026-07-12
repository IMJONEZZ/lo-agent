// lo-jlens LD_PRELOAD shim — live J-space steering of the user's OWN llama-server.
//
// Activation access needs the ggml eval callback registered at context
// creation. A stock, already-running llama-server can't be attached to. This
// shim interposes `llama_init_from_model` in a dynamically-linked llama-server:
// it injects an eval callback that reads a steering file and adds steering
// vectors to the residual stream (`l_out-<il>`) mid-graph, so the tokens the
// server actually generates are steered. One env var + one restart; no fork of
// llama.cpp, public API only.
//
//   LD_PRELOAD=/path/lo_jlens_shim.so LO_JLENS_STEER=/path/steer.bin \
//       llama-server -m model.gguf ...        # same flags as before
//
// steer.bin format (little-endian):
//   magic "LJS1" | u32 n_edits |
//   repeat n_edits: i32 layer, i32 pos_start, i32 pos_end(-1=∞), i32 d, d*f32
// The file is re-read (mtime-gated) each decode, so an operator can update the
// steering set live without restarting. Only additive edits (steer / negative-
// steer / suppress) are supported here; ablate/swap use the sidecar or exports.
//
// SPIKE-STAGE: compiles/links against libllama's public API. Live interposition
// is gated on the target being a dynamically-linked llama-server (see
// `lo lens doctor`). Not yet validated against a running server in this tree.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

struct Edit {
    int layer;
    int pos_start;
    int pos_end;      // -1 = unbounded
    std::vector<float> vec;  // length d_model
};

std::mutex g_mu;
std::vector<Edit> g_edits;
time_t g_mtime = 0;
const char * g_path = nullptr;

int parse_l_out(const char * name) {
    const char * p = strstr(name, "l_out-");
    if (!p) return -1;
    if (p != name && p[-1] != '#') return -1;
    char * end = nullptr;
    long il = strtol(p + 6, &end, 10);
    if (end == p + 6) return -1;
    if (*end != '\0' && *end != '#') return -1;
    return (int) il;
}

void reload_if_changed() {
    if (!g_path) return;
    struct stat st;
    if (stat(g_path, &st) != 0) return;
    if (st.st_mtime == g_mtime) return;
    FILE * f = fopen(g_path, "rb");
    if (!f) return;
    char magic[4];
    uint32_t n = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "LJS1", 4) != 0 ||
        fread(&n, 4, 1, f) != 1) { fclose(f); return; }
    std::vector<Edit> edits;
    for (uint32_t i = 0; i < n; i++) {
        int32_t hdr[4];
        if (fread(hdr, 4, 4, f) != 4) break;
        Edit e{hdr[0], hdr[1], hdr[2], std::vector<float>(hdr[3])};
        if (fread(e.vec.data(), 4, hdr[3], f) != (size_t) hdr[3]) break;
        edits.push_back(std::move(e));
    }
    fclose(f);
    std::lock_guard<std::mutex> lk(g_mu);
    g_edits.swap(edits);
    g_mtime = st.st_mtime;
    fprintf(stderr, "[lo-jlens] loaded %zu steering edit(s) from %s\n", g_edits.size(), g_path);
}

bool eval_cb(struct ggml_tensor * t, bool ask, void * /*user*/) {
    const int il = parse_l_out(t->name);
    if (ask) return il >= 0;           // request only l_out nodes
    if (il < 0) return true;
    reload_if_changed();
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_edits.empty()) return true;
    if (t->type != GGML_TYPE_F32) return true;
    const int64_t d = t->ne[0];
    const int64_t n = t->ne[1];
    float * data = (float *) t->data;
    for (const auto & e : g_edits) {
        if (e.layer != il || (int64_t) e.vec.size() != d) continue;
        const int64_t hi = e.pos_end < 0 ? n : (e.pos_end < n ? e.pos_end : n);
        for (int64_t p = e.pos_start; p < hi; p++)
            for (int64_t j = 0; j < d; j++)
                data[p * d + j] += e.vec[j];
    }
    return true;
}

}  // namespace

extern "C" struct llama_context * llama_init_from_model(
        struct llama_model * model, struct llama_context_params params) {
    static auto real = (decltype(&llama_init_from_model))
        dlsym(RTLD_NEXT, "llama_init_from_model");
    if (!g_path) {
        g_path = getenv("LO_JLENS_STEER");
        fprintf(stderr, "[lo-jlens] shim active%s%s\n",
                g_path ? "; steering from " : " (set LO_JLENS_STEER to steer)",
                g_path ? g_path : "");
    }
    if (g_path && !params.cb_eval) {         // don't clobber a real callback
        params.cb_eval = eval_cb;
        params.cb_eval_user_data = nullptr;
    }
    return real(model, params);
}
