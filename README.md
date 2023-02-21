# GPT3 embeddings movies search
This repository relates to an article [Leveraging GPT-3 for Search Solutions Development](https://blog.weblab.technology/leveraging-gpt-3-for-search-solutions-development-2b6b2a7b29a9).

## TMDB 5000 movies embeddings GPT-3 search solution

`search_movies.ipynb` contains a notebook with steps to build a primitive search solution with [GPT-3 embeddings API](https://beta.openai.com/docs/guides/embeddings/what-are-embeddings) for [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). This repository includes pre-indexed embeddings created with the babbage model family.

## **Choosing the right model**
The conversion of document text into a vector representative can happen in the background, while the vectorization of the search query should occur during runtime. There are several GPT-3 model families that OpenAI offers: 

```
text-search-ada-doc-001: 1024
text-search-babbage-doc-001: 2048
text-search-curie-doc-001: 4096
text-search-davinci-doc-001: 12288
```  

Higher vector dimensions lead to more embedded information and, thus, also higher costs and slower searches.

Documents are usually long and queries generally short and incomplete. Therefore, the vectorization of any document differs significantly from any query's vectorization considering the content's density and size. OpenAI know that, and so they offer two paired models, ```-doc``` and ```-query```:

```
text-search-ada-query-001: 1024
text-search-babbage-query-001: 2048
text-search-curie-queryc-001: 4096
text-search-davinci-query-001: 12288
```  

It is important to note that the query and document must both utilize the same model family and have the same length of the output vector.

## **Example dataset**
It may be easiest to observe and understand the power of this search solution through example. For this example, let us draw on the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/code) which contains metadata about roughly 5,000 movies from [TMDb](https://www.themoviedb.org/). We will build a search solution relying only on movie documents, not reviews.

The dataset contains plenty of columns, but our vectorization process will be built around the title and overview columns only.

Example of the record:

```
Title: Harry Potter and the Half-Blood Prince
Overview: As Harry begins his sixth year at Hogwarts, he discovers an old book marked as 'Property of the Half-Blood Prince', and begins to learn more about Lord Voldemort's dark past.
```  

Let’s map the dataset into ready-to-indexing text:

``` Python
datafile_path = "./tmdb_5000_movies.csv"
df = pd.read_csv(datafile_path)

def combined_info(row):
  columns = ['title', 'overview']
  columns_to_join = [f"{column.capitalize()}: {row[column]}" for column in columns]
  return '\n'.join(columns_to_join)
  
df['combined_info'] = df.apply(lambda row: combined_info(row), axis=1)
```

The embedding process is straightforward:

``` Python
def get_embedding(text, model="text-search-babbage-doc-001"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

get_embedding(df['combined_info'][0])
```

This block of code outputs a list the size of which is equal to the parameters that the model is operating with, which in the case of ```text-search-babbage-doc-001``` is 2048.

A similar embedding process should be applied to all documents we would like to search on:

``` Python
df['combined_info_search'] = df['combined_info'].apply(lambda x: get_embedding(x, model='text-search-babbage-doc-001'))
df.to_csv('./tmdb_5000_movies_search.csv', index=False)
```

Column ```combined_info_search``` will hold a vector representation of the combined_text.

And, surprisingly, that’s already it! Finally, we are ready to perform a sample search query:

``` Python
from openai.embeddings_utils import get_embedding, cosine_similarity

def search_movies(df, query, n=3, pprint=True):
    embedding = get_embedding(
        query,
        engine="text-search-babbage-query-001"
    )
    df["similarities"] = df.combined_info_search.apply(lambda x: cosine_similarity(x, embedding))

    res = (
        df.sort_values("similarities", ascending=False)
        .head(n)
        .combined_info
    )
    if pprint:
        for r in res:
            print(r[:200])
            print()
    return res


res = search_movies(df, "movie about the wizardry school", n=3)
```

### These are the results we get:

```
Title: Harry Potter and the Philosopher's Stone
Overview: Harry Potter has lived under the stairs at his aunt and uncle's house his whole life. But on his 11th birthday, he learns he's a powerful wizard -- with a place waiting for him at the Hogwarts School of Witchcraft and Wizardry. As he learns to harness his newfound powers with the help of the school's kindly headmaster, Harry uncovers the truth about his parents' deaths -- and about the villain who's to blame.

Title: Harry Potter and the Goblet of Fire
Overview: Harry starts his fourth year at Hogwarts, competes in the treacherous Triwizard Tournament and faces the evil Lord Voldemort. Ron and Hermione help Harry manage the pressure – but Voldemort lurks, awaiting his chance to destroy Harry and all that he stands for.

Title: Harry Potter and the Prisoner of Azkaban
Overview: Harry, Ron and Hermione return to Hogwarts for another magic-filled year. Harry comes face to face with danger yet again, this time in the form of an escaped convict, Sirius Black – and turns to sympathetic Professor Lupin for help.
```  

The overview for ‘Harry Potter and the Philosopher's Stone’ contains the words ‘wizardry’ and ‘school’ and comes first in the search output. The second result no longer contains the word ‘school’, but still retains words close to ‘wizardry’, ‘Triwizard’. The third result contains only a synonym of ‘wizardry’ – magic.

There are, of course, a multitude of other movies within this database that feature schools or wizards (or both), but the above were the only ones returned to us. This is clear proof that the search solution works and actually understood the context of our query.

We used the babbage model with only 2048 parameters. Davinci has six times more (12,288) parameters and can, thus, perform significantly better with regard to highly complex queries.

The search solution may occasionally fail to produce output relevant to some queries. For instance, the query ‘movies about wizards in school’ produces:

```
Title: Harry Potter and the Philosopher's Stone
Overview: Harry Potter has lived under the stairs at his aunt and uncle's house his whole life. But on his 11th birthday, he learns he's a powerful wizard -- with a place waiting for him at the Hogwarts School of Witchcraft and Wizardry. As he learns to harness his newfound powers with the help of the school's kindly headmaster, Harry uncovers the truth about his parents' deaths -- and about the villain who's to blame.

Title: Dumb and Dumberer: When Harry Met Lloyd
Overview: This wacky prequel to the 1994 blockbuster goes back to the lame-brained Harry and Lloyd's days as classmates at a Rhode Island high school, where the unprincipled principal puts the pair in remedial courses as part of a scheme to fleece the school.

Title: Harry Potter and the Prisoner of Azkaban
Overview: Harry, Ron and Hermione return to Hogwarts for another magic-filled year. Harry comes face to face with danger yet again, this time in the form of an escaped convict, Sirius Black – and turns to sympathetic Professor Lupin for help.
```

What is ‘Dumb and Dumberer: When Harry Met Lloyd’ doing here you may wonder? Thankfully, this issue was not reproduced on parameters with more parameters.

## Calculating the distance between the query and documents
The search output should consist of documents sorted in descending order by relevance. To achieve this, we should be aware of the distance between the current query and each document. The shorter the length, the comparatively more relevant the output. Then, after a defined maximum reach, we should stop considering the relevance of the remaining documents.

In the aforementioned example, we used [cosine similarity](https://www.sciencedirect.com/topics/computer-science/cosine-similarity) to calculate distance due to the high dimensionality of the vector space. With the babbage model, we have 2048 parameters.

Distance calculation algorithms tend to represent this similarity (difference) between a query and a document with a single number. However, we cannot rely on the [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) because of the [curse of dimensionality](https://link.springer.com/referenceworkentry/10.1007/978-0-387-39940-9_133) – distances will be too similar. This is because Euclidean distance becomes highly impractical beyond around seven dimensions – at this point, distances fall into the same buckets and become almost identical.
 
If you would like, you may check out the resulting repository [here.](https://github.com/weblab-technology/gpt3-embeddings-movies-search)
Alternatively, you can play with it in Google Colab [here.](https://colab.research.google.com/drive/1ncZRUnicBWtsCeaRBYwh6kJvyEGSYAmz?usp=sharing)

## Time complexity
We used the brute force approach to sort the documents. Let’s determine:
- n: number of points in the training dataset
- d: data dimensionality

The search time complexity for brute force solutions is _O(n * d * n * log(n))_. Parameter **d** depends on the model (in the case of babbage, it is equal to 2048) while we have _O(nlog(n))_ block due to the sorting step.

It is critical to remind ourselves at this stage that smaller models are faster and cheaper. For instance, in the search case similarity calculation step, the Ada model is two times faster, while the Davinci model is six times slower.

Cosine similarity calculations between the query and 4803 documents of 2048 dimensions took 1260ms on my M1 Pro. In the current implementation, time required to calculate would grow linearly to the total number of documents. Simultaneously, this approach supports computation parallelization.

## Alternatives to the brute force solution
In search solutions, queries should be completed as quickly as possible. And this price is usually paid on the side of training and pre-caching time. We can use data structures like a k-d tree, r-tree, or ball tree. Consider the article from _Towards Data Science_ about the computational complexity analysis of these methods: They all lead to computational complexity close to _O(k * log(n))_, where **k** is the number of elements we would like to return within a single batch. 

K-d trees, ball trees, and r-trees constitute data structures that are used to store and efficiently search for points in N-dimensional space, such as our meaning vectors.

K-d and ball trees are tree-based data structures that use an iterative, binary partitioning scheme to divide the space into regions, with each node in the tree representing a subregion. K-d trees are particularly efficient at searching for points within a specific range or finding the nearest neighbor to a given point. 

Similarly, r-trees are also used to store points in N-dimensional space, however, they are much more efficient at searching for points within a specific region or finding all points within a certain distance of a given point. Importantly, r-trees use a different partitioning scheme to k-d trees and ball trees; they divide the space into "rectangles" rather than binary partitions.

Tree implementations fall outside the scope of this article, and different implementations will lead to different search outputs.

## Query time
Perhaps, the most significant disadvantage of the current search solution is that we must call an external OpenAI API to retrieve the query embedding vector. No matter how quickly our algorithm is able to find the nearest neighbors, a sequential blocking step will be required.

**Text-search-babbage-query-001**
```
Number of dimensions: 2048
Number of queries: 100
Average duration: 225ms
Median duration: 207ms
Max duration: 1301ms
Min duration: 176ms
```
**Text-search-ada-query-002**
```
Number of dimensions: 1536
Number of queries: 100
Average duration: 264ms
Median duration: 250ms
Max duration: 859ms
Min duration: 215ms
```
**Text-search-davinci-query-001**
```
Number of dimensions: 12288
Number of queries: 100
Average duration: 379ms
Median duration: 364ms
Max duration: 1161ms
Min duration: 271ms
```
 
If we take the median as a referencing point, we can see that ada-002 is +28% slower and davinci-001 is +76% slower.

## Limitations of GPT-3 Search Embedding
Referring to [Nils Reimer's article about dense text embeddings model comparison](https://medium.com/@nils_reimers/openai-gpt-3-text-embeddings-really-a-new-state-of-the-art-in-dense-text-embeddings-6571fe3ec9d9), we may conclude that GPT-3 does not provide an exceptional performance or output quality and requires dependence on external API that is rather slow. GPT-3 has the capability to support inputs of up to 4096 tokens (approximately 3072 words), however, there is no truncation service available through the API and attempting to encode text longer than 4096 tokens will result in an error. Thus, it is the responsibility of the user – you – to determine how much text can actually be encoded.

Also, training costs are relatively high with OpenAI! 
Alternatively, you may consider trying [TAS-B](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b) or [multi-qa-mpnet-base-dot-v1](http://multi-qa-mpnet-base-dot-v1/).
