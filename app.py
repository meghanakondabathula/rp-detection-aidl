import gradio as gr

def hello(name):
    return "Hello " + name

demo = gr.Interface(
    fn=hello,
    inputs=gr.Textbox(),
    outputs=gr.Textbox()
)

if __name__ == "__main__":
    demo.launch()