import gradio as gr
import requests
import os

API = os.getenv("DOC_RAG_API", "http://127.0.0.1:8000")


def ingest_file(file, retriever, do_embed):
    """Call POST /ingest."""
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
    """Call GET /retrieve and return a list of {page_number,score}."""
    params = {"doc_id": doc_id, "query": query, "retriever": retriever, "top_k": top_k}
    r = requests.get(f"{API}/retrieve", params=params)
    r.raise_for_status()
    results = r.json()["results"]
    # Gradio can consume a list-of-dicts directly into a DataFrame component
    return results


def generate_text(query, generator, system_prompt, prompt_template):
    """Call POST /generate for text-only generation."""
    body = {
        "query": query,
        "generator": generator,
        "system_prompt": system_prompt or None,
        "prompt_template": prompt_template or None,
    }
    r = requests.post(f"{API}/generate", json=body)
    r.raise_for_status()
    return r.json()["answer"]


def rag_qa(doc_id, query, retriever, generator, top_k, system_prompt, prompt_template):
    """Call POST /rag for end-to-end RAG."""
    body = {
        "doc_id": doc_id,
        "query": query,
        "retriever": retriever,
        "generator": generator,
        "top_k": top_k,
        "system_prompt": system_prompt or None,
        "prompt_template": prompt_template or None,
    }
    r = requests.post(f"{API}/rag", json=body)
    r.raise_for_status()
    js = r.json()
    return js["retrieval_results"], js["answer"]


with gr.Blocks(title="DocRAG Demo UI") as demo:
    gr.Markdown("# 📄 DocRAG Demo")
    with gr.Tab("Ingest"):
        with gr.Row():
            pdf_in = gr.File(label="Upload PDF", file_types=[".pdf"])
            retriever = gr.Dropdown(
                ["colnomic-3b", "colpali-v1.3"], value="colnomic-3b", label="Retriever"
            )
            do_embed = gr.Checkbox(value=True, label="Build FAISS index now")
            ingest_btn = gr.Button("Ingest")
        with gr.Row():
            doc_id_out = gr.Textbox(label="Document ID")
            num_pages_out = gr.Number(label="Pages")
            paths_out = gr.Textbox(label="Page paths", lines=3)
        ingest_btn.click(
            ingest_file,
            inputs=[pdf_in, retriever, do_embed],
            outputs=[doc_id_out, num_pages_out, paths_out],
        )

    with gr.Tab("Retrieve"):
        with gr.Row():
            rid = gr.Textbox(label="Document ID")
            query_r = gr.Textbox(label="Query")
        with gr.Row():
            retr2 = gr.Dropdown(
                ["colnomic-3b", "colpali-v1.3"], value="colnomic-3b", label="Retriever"
            )
            top_k_r = gr.Slider(1, 50, value=5, step=1, label="Top k")
            retrieve_btn = gr.Button("Retrieve")
        results_df = gr.DataFrame(
            headers=["page_number", "score"], label="Top pages & scores"
        )
        retrieve_btn.click(
            retrieve_pages,
            inputs=[rid, query_r, retr2, top_k_r],
            outputs=results_df,
        )

    with gr.Tab("Generate"):
        with gr.Row():
            query_g = gr.Textbox(label="Query")
            gen_model = gr.Dropdown(
                ["internvl-3b", "qwen2-vl-chat"], value="internvl-3b", label="Generator"
            )
        system_prompt = gr.Textbox(label="System prompt (optional)", lines=2)
        prompt_template = gr.Textbox(label="Prompt template (optional)", lines=2)
        generate_btn = gr.Button("Generate")
        answer_out = gr.Textbox(label="Answer", lines=5)
        generate_btn.click(
            generate_text,
            inputs=[query_g, gen_model, system_prompt, prompt_template],
            outputs=answer_out,
        )

    with gr.Tab("RAG"):
        with gr.Row():
            rag_doc = gr.Textbox(label="Document ID")
            query_rag = gr.Textbox(label="Query")
        with gr.Row():
            retr3 = gr.Dropdown(
                ["colnomic-3b", "colpali-v1.3"], value="colnomic-3b", label="Retriever"
            )
            gen3 = gr.Dropdown(
                ["internvl-3b", "qwen2-vl-chat"], value="internvl-3b", label="Generator"
            )
            top_k_rag = gr.Slider(1, 50, value=5, step=1, label="Top k")
        system_prompt2 = gr.Textbox(label="System prompt (optional)", lines=2)
        prompt_template2 = gr.Textbox(label="Prompt template (optional)", lines=2)
        rag_btn = gr.Button("Run RAG")
        rag_results_df = gr.DataFrame(
            headers=["page_number", "score"], label="Retrieval results"
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

demo.launch()
