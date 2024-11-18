<h1 align="center">Blog Generation Using LLAMA 💃</h1>

This project explores the use of LLAMA, a powerful language model, to generate high-quality blog posts on a variety of topics. The goal is to create an automated system that can write blog content in a natural, human-like style, tailored to specific subjects or keywords. By leveraging LLAMA's advanced natural language processing capabilities, the system can produce coherent, engaging, and informative blog posts with minimal input.

![Blog Generation](blog_generation_preview.jpg?raw=true)

## Getting Started 🏁

> Make sure your system has [`CUDA`](https://developer.nvidia.com/cuda-downloads), [`conda`](https://anaconda.org/anaconda/conda) and [`git`](https://git-scm.com/) installed.

```bash
git clone https://github.com/MahmoudHesham/Blog-Generation-Using-LLAMA
cd Blog-Generation-Using-LLAMA

# create environment using conda
conda create -n llm_blog python=3.9
conda activate llm_blog
```

## Download Model ⚙
* You can download LLAMA model (e.g. **llama-2-7b-chat.ggmlv3.q8_0.bin**) either from [`LLAMA`](https://www.llama.com) or [`Hugging Face`](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main).
* Place model in the following path: ```Blog-Generation-Using-LLAMA/models```

## Start application 🚀
* Open the command line and launch streamlit server by using the command ```streamlit run app.py```.
* This will open your web browser with the user interface where you can create blog posts.

## License
* All the tutorials, scripts, tools and software-bridges developed by VFX Kitchen. are free for academic and non-commercial use only.

## Support [VFX Kitchen.](https://www.youtube.com/VFXKitchen) ❤
* support the author by subscribing to the youtube channel, liking the videos and share it with your VFX and AI enthusiasts friends.
