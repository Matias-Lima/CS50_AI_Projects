import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # The number of page is equal to the size of the corpus
    n_pages = len(corpus)

    # Check if the page has outgoing links
    if len(corpus[page]) == 0:
        return {page_name: 1 / n_pages for page_name in corpus}

    # Calculate the probability of picking any page at random
    r_prob = (1 - damping_factor) / n_pages

    # Calculate the probability of picking a link from the current page
    l_prob = damping_factor / len(corpus[page])

    # Initialize the probability distribution dictionary with random probabilities
    p_dist = {page_name: r_prob for page_name in corpus}

    # Add link probabilities to the distribution
    for linked_page in corpus[page]:
        p_dist[linked_page] += l_prob


    return p_dist

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    def normalize_visits(access, n):
        return {page_name: (access_num / n) for page_name, access_num in access.items()}


    # Inicializar o dicionário 'visits' com um loop for
    access = {}
    for page_name in corpus:
        access[page_name] = 0

    atual_page = random.choice(list(access.keys()))
    access[atual_page] += 1


    # Loop para as próximas n-1 amostras
    for i in range(n - 1):

        trans_model = transition_model(corpus, atual_page, damping_factor)
        page_list, probabilities = zip(*trans_model.items())

        rand_val = random.random()
        t_prob = 0

        for j, probability in enumerate(probabilities):
            t_prob += probability
            if rand_val <= t_prob:
                atual_page = page_list[j]
                break

        access[atual_page] += 1

    # Normalizar as visitas usando o número de amostras (n)
    page_ranks = normalize_visits(access, n)

    return page_ranks

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    max_iterations = 10000
    convergence_threshold = 0.001

    num_pages = len(corpus)
    initial_rank = 1 / num_pages
    random_prob = (1 - damping_factor) / num_pages
    ranks = {page: initial_rank for page in corpus}
    iteration = 0
    max_rank_change = convergence_threshold + 1  # set an initial value greater than the threshold

    while max_rank_change > convergence_threshold and iteration < max_iterations:
        new_ranks = {}
        for page in corpus:
            surfing_prob = sum(ranks[other_page] / len(corpus[other_page]) for other_page in corpus if
                               page in corpus[other_page]) + sum(
                ranks[other_page] * initial_rank for other_page in corpus if not corpus[other_page])
            new_rank = random_prob + damping_factor * surfing_prob
            new_ranks[page] = new_rank

        normalization_factor = sum(new_ranks.values())
        new_ranks = {page: rank / normalization_factor for page, rank in new_ranks.items()}

        max_rank_change = max(abs(new_ranks[page] - ranks[page]) for page in corpus)
        ranks = new_ranks
        iteration += 1

    print(f"Number of iterations to converge: {iteration} ")

    return ranks

if __name__ == "__main__":
    print(crawl("corpus0"))
    main()


