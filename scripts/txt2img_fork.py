from ldm.modules.encoders.modules import FrozenCLIPEmbedder
from ldm.modules.embedding_manager import EmbeddingManager

def main():
    clip = FrozenCLIPEmbedder()
    embedding_manager = EmbeddingManager(
        placeholder_strings=['*'],
        initializer_words=['plush', 'doll'],
        per_image_tokens=False,
        num_vectors_per_token=6,
        progressive_words=False,
        embedder=clip
    ).to('mps')

    embedding_manager.load('/Users/birch/git/stable-diffusion/logs/2022-09-20T01-49-11_fumo/checkpoints/embeddings.pt')

    clip.encode('', embedding_manager=embedding_manager)


if __name__ == '__main__':
    main()
