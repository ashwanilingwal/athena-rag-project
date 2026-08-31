# Chunking

## Chunk size
Chunk size is how much text goes into one embedding. Small chunks produce focused
vectors but lose surrounding context. Large chunks preserve context but dilute the
embedding, because a single vector must represent several unrelated ideas at once.
A common starting point is five hundred and twelve tokens.

## Chunk overlap
Chunk overlap repeats the tail of one chunk at the head of the next so that an idea
split across a boundary still appears intact somewhere. Overlap is usually ten to
fifteen percent of the chunk size.

## Structure beats size
Splitting on document structure matters more than the exact token count. Splitting
on headings and paragraph boundaries, and prefixing each chunk with its heading
path, reliably beats fixed-width splitting on the same corpus.
