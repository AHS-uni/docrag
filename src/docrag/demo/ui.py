import os
import requests
import gradio as gr

API = os.getenv("DOC_RAG_API", "http://127.0.0.1:8000")


# ──────────────────────────────
# Back-end helpers
# ──────────────────────────────
def refresh_models():
    r = requests.get(f"{API}/models")
    r.raise_for_status()
    return [[m["name"], m["type"], m["loaded"]] for m in r.json()["models"]]


def _post_model(endpoint, name, mtype):
    params = {"name": name, "type": mtype}
    r = requests.post(f"{API}/models/{endpoint}", params=params)
    r.raise_for_status()
    return refresh_models()


def load_model(name, mtype):
    return _post_model("load", name, mtype)


def unload_model(name, mtype):
    return _post_model("unload", name, mtype)


def download_model(name, mtype):
    params = {"name": name, "type": mtype}
    r = requests.post(f"{API}/models/download", params=params)
    r.raise_for_status()
    return f"Snapshot cached at:\n{r.json()['path']}"


def ingest_file(file, retriever, do_embed):
    """POST /ingest → (doc_id, num_pages, paths_str)."""
    if file is None:
        return gr.update(), gr.update(), gr.update()

    files = {
        "file": (os.path.basename(file.name), open(file.name, "rb"), "application/pdf")
    }
    params = {"retriever": retriever, "do_embed": do_embed}
    r = requests.post(f"{API}/ingest", params=params, files=files)
    r.raise_for_status()
    j = r.json()
    return j["doc_id"], j["num_pages"], "\n".join(j["page_paths"])


def retrieve_pages(doc_id, query, retriever, top_k):
    """GET /retrieve → rows for gr.DataFrame."""
    params = {
        "doc_id": doc_id,
        "query": query,
        "retriever": retriever,
        "top_k": int(top_k),
    }
    r = requests.get(f"{API}/retrieve", params=params)
    r.raise_for_status()
    return [[row["page_number"], row["score"]] for row in r.json()["results"]]


def generate_text(doc_id, pages_str, query, generator, system_prompt, prompt_template):
    """POST /generate (supports doc_id & pages)."""
    pages = None
    if pages_str:
        pages = [int(p.strip()) for p in pages_str.split(",") if p.strip()]

    body = {
        "doc_id": doc_id or None,
        "pages": pages,
        "query": query,
        "generator": generator,
        "system_prompt": system_prompt or None,
        "prompt_template": prompt_template or None,
    }
    r = requests.post(f"{API}/generate", json=body)
    r.raise_for_status()
    return r.json()["answer"]


def rag_qa(doc_id, query, retriever, generator, top_k, system_prompt, prompt_template):
    """POST /rag → (rows, answer)."""
    body = {
        "doc_id": doc_id,
        "query": query,
        "retriever": retriever,
        "generator": generator,
        "top_k": int(top_k),
        "system_prompt": system_prompt or None,
        "prompt_template": prompt_template or None,
    }
    r = requests.post(f"{API}/rag", json=body)
    r.raise_for_status()
    js = r.json()
    rows = [[row["page_number"], row["score"]] for row in js["retrieval_results"]]
    return rows, js["answer"]


# ──────────────────────────────
# Gradio UI
# ──────────────────────────────
with gr.Blocks(title="DocRAG Demo UI") as demo:
    gr.Markdown("# 📄 DocRAG Demo")

    with gr.Tab("Models"):
        model_table = gr.DataFrame(
            headers=["name", "type", "loaded"],
            datatype=["str", "str", "bool"],
            interactive=False,
            label="Registered models",
        )

        refresh_btn = gr.Button("🔄 Refresh list")

        with gr.Row():
            sel_name = gr.Textbox(label="Model key (e.g. colpali)")
            sel_type = gr.Dropdown(
                ["retriever", "generator"], value="retriever", label="Type"
            )

        with gr.Row():
            load_btn = gr.Button("Load")
            unload_btn = gr.Button("Unload")
            download_btn = gr.Button("Download snapshot")

        download_msg = gr.Textbox(label="Download status / path", lines=2)

        # Wire up callbacks
        refresh_btn.click(refresh_models, outputs=model_table)
        load_btn.click(load_model, inputs=[sel_name, sel_type], outputs=model_table)
        unload_btn.click(unload_model, inputs=[sel_name, sel_type], outputs=model_table)
        download_btn.click(
            download_model, inputs=[sel_name, sel_type], outputs=download_msg
        )

        # populate on launch
        demo.load(refresh_models, None, model_table)

    # ─── Ingest ────────────────────────────────────────────────
    with gr.Tab("Ingest"):
        with gr.Row():
            pdf_in = gr.File(label="Upload PDF", file_types=[".pdf"])
            retriever_ing = gr.Dropdown(
                ["colpali", "colqwen"], value="colpali", label="Retriever"
            )
            do_embed = gr.Checkbox(value=True, label="Build FAISS index now")
            ingest_btn = gr.Button("Ingest")

        with gr.Row():
            doc_id_out = gr.Textbox(label="Document ID")
            num_pages_out = gr.Number(label="Pages", interactive=False)
            paths_out = gr.Textbox(label="Page paths", lines=3)

        ingest_btn.click(
            ingest_file,
            inputs=[pdf_in, retriever_ing, do_embed],
            outputs=[doc_id_out, num_pages_out, paths_out],
        )

    # ─── Retrieve ──────────────────────────────────────────────
    with gr.Tab("Retrieve"):
        with gr.Row():
            rid = gr.Textbox(label="Document ID")
            query_r = gr.Textbox(label="Query")
        with gr.Row():
            retr2 = gr.Dropdown(
                ["colpali", "colqwen"], value="colpali", label="Retriever"
            )
            top_k_r = gr.Number(value=3, minimum=1, precision=0, label="Top k")
            retrieve_btn = gr.Button("Retrieve")
        results_df = gr.DataFrame(
            headers=["page_number", "score"],
            label="Top pages & scores",
            datatype=["number", "number"],
            interactive=False,
        )
        retrieve_btn.click(
            retrieve_pages,
            inputs=[rid, query_r, retr2, top_k_r],
            outputs=results_df,
        )

    # ─── Generate (w/ optional doc pages) ──────────────────────
    with gr.Tab("Generate"):
        with gr.Row():
            gen_doc = gr.Textbox(label="Document ID (optional)")
            gen_pages = gr.Textbox(
                label="Pages comma-separated (optional)",
                placeholder="e.g. 0, 3, 7",
            )
        with gr.Row():
            query_g = gr.Textbox(label="Query")
            gen_model = gr.Dropdown(
                ["internvl", "qwenvl"], value="qwenvl", label="Generator"
            )
        # defaults requested by user
        default_sys_prompt = (
            "You are a helpful vision-language assistant. You will be shown an "
            "image and a user question. Your task is to answer the question as "
            "briefly and accurately as possible using only the information "
            "visible in the image. Do not add explanations or extra details. "
            "If the answer is not present in the provided document or unclear, "
            "respond with 'Not Answerable'."
        )
        default_template = "Question: {text}. Answer:"

        system_prompt = gr.Textbox(
            label="System prompt", lines=3, value=default_sys_prompt
        )
        prompt_template = gr.Textbox(
            label="Prompt template", lines=2, value=default_template
        )
        generate_btn = gr.Button("Generate")
        answer_out = gr.Textbox(label="Answer", lines=5)

        generate_btn.click(
            generate_text,
            inputs=[
                gen_doc,
                gen_pages,
                query_g,
                gen_model,
                system_prompt,
                prompt_template,
            ],
            outputs=answer_out,
        )

    # ─── RAG ───────────────────────────────────────────────────
    with gr.Tab("RAG"):
        with gr.Row():
            rag_doc = gr.Textbox(label="Document ID")
            query_rag = gr.Textbox(label="Query")
        with gr.Row():
            retr3 = gr.Dropdown(
                ["colpali", "colqwen"], value="colpali", label="Retriever"
            )
            gen3 = gr.Dropdown(
                ["internvl", "qwenvl"], value="qwenvl", label="Generator"
            )
            top_k_rag = gr.Number(value=3, minimum=1, precision=0, label="Top k")
        system_prompt2 = gr.Textbox(
            label="System prompt", lines=3, value=default_sys_prompt
        )
        prompt_template2 = gr.Textbox(
            label="Prompt template", lines=2, value=default_template
        )
        rag_btn = gr.Button("Run RAG")
        rag_results_df = gr.DataFrame(
            headers=["page_number", "score"],
            label="Retrieval results",
            datatype=["number", "number"],
            interactive=False,
        )
        rag_answer_out = gr.Textbox(label="Answer", lines=5)

        rag_btn.click(
            rag_qa,
            inputs=[
                rag_doc,
                query_rag,
                retr3,
                gen3,
                top_k_rag,
                system_prompt2,
                prompt_template2,
            ],
            outputs=[rag_results_df, rag_answer_out],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
