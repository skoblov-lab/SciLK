import numpy as np

from infgain.preprocessing import TILDA_SEP, ngrams_from_dict
from word2gauss.words import Vocabulary


def gaussian_representation(token: str, vocab_gauss, embed) -> tuple:
    """
    Gaussian representation of `token`.
    :param token:
    :param vocab_gauss:
    :param embed:
    :return: list of mus, list of sigmas
    """
    # In case of tilda-token sort the words.
    if TILDA_SEP in token:
        token = token.split(TILDA_SEP)
        token = sorted(token)
        token = TILDA_SEP.join(token)

    mu = embed.mu[vocab_gauss.word2id(token)]
    sigma = np.diag(embed.sigma[vocab_gauss.word2id(token)])
    return mu, sigma


def gaussian_kl(gaus1: dict, gaus2: dict) -> float:
    """
    Calculates KL-divergence between gaussians `gaus1` and `gaus2`.
    :param gaus1:
    :param gaus2:
    :return:
    """
    divergence = (1. / 2.) * \
                 (np.trace(np.dot(np.linalg.inv(gaus2['sigma']), gaus1['sigma'])) +
                  np.dot(gaus2['mu'] - gaus1['mu'], np.dot(np.linalg.inv(gaus2['sigma']), gaus2['mu'] - gaus1['mu'])) -
                  len(gaus1['mu']) + np.log(np.linalg.det(gaus2['sigma']) / np.linalg.det(gaus1['sigma'])))
    return divergence


def calculate_closed_kl(ngram: str, vocab_gauss, embed, order=0) -> float:
    """
    Calculate KL-divergence.
    :param ngram: string
    :param vocab_gauss:
    :param embed:
    :param order: int
        If order is 0, then KL(ngram, tilda-token) is calculated;
        if order is 1, then KL(tilda-token, ngram) is calculated.
    """
    # Gaussian representation of `ngram`.
    mu_ngram, sigma_ngram = gaussian_representation(ngram, vocab_gauss, embed)

    # Gaussian representation of tilda-token.
    tilda = ngram.replace('_', '~')
    mu_tilda, sigma_tilda = gaussian_representation(tilda, vocab_gauss, embed)

    if order == 0:
        gaus1 = {'mu': mu_ngram, 'sigma': sigma_ngram}
        gaus2 = {'mu': mu_tilda, 'sigma': sigma_tilda}
    elif order == 1:
        gaus1 = {'mu': mu_tilda,  'sigma': sigma_tilda}
        gaus2 = {'mu': mu_ngram,  'sigma': sigma_ngram}
    else:
        raise RuntimeError

    return gaussian_kl(gaus1, gaus2)


def calculate_variational_kl(ngram: str, vocab_gauss, embed, order=0):
    """
    Calculate variational KL-divergence of `ngram` and corresponding mixture.
    :param ngram: string
    :param vocab_gauss:
    :param embed:
    :param order: int
        If order is 0, then KL(ngram, mixture) is calculated;
        if order is 1, then KL(mixture, ngram) is calculated.
    """

    # Gaussian representation for every word of `ngram`.
    mu_words = []
    sigma_words = []
    for word in ngram.split('_'):
        mu, sigma = gaussian_representation(word, vocab_gauss, embed)
        mu_words.append(mu)
        sigma_words.append(sigma)
    mu_words = np.array(mu_words)
    sigma_words = np.array(sigma_words)

    # Gaussian representation of `ngram`.
    mu_ngram, sigma_ngram = gaussian_representation(ngram, vocab_gauss, embed)
    mu_ngram = np.array([mu_ngram])
    sigma_ngram = np.array([sigma_ngram])

    # len(ngram.split('_') times of (1. / len(ngram.split('_')))
    # TODO: Check whether it's correct!
    weights_words = [1. / len(ngram.split('_'))] * len(ngram.split('_'))
    weights_ngram = [1.]

    def complex_kl(f: dict, g: dict) -> float:
        """
        Calculate KL-divergence between words of `ngram` and tilda-token.
        :param f:
        :param g:
        :return:
        """
        ans = 0
        for a, pi in enumerate(f['weights']):
            sum1 = 0
            sum2 = 0

            for b, ksi in enumerate(f['weights']):
                params_a = {'mu': f['mus'][a], 'sigma': f['sigmas'][a]}
                params_b = {'mu': f['mus'][b], 'sigma': f['sigmas'][b]}
                sum1 += ksi * np.exp((-1.) * gaussian_kl(params_a, params_b))

            for c, omega in enumerate(g['weights']):
                params_a = {'mu': f['mus'][a], 'sigma': f['sigmas'][a]}
                params_c = {'mu': g['mus'][c], 'sigma': g['sigmas'][c]}
                sum2 += omega * np.exp((-1) * gaussian_kl(params_a, params_c))

            ans += pi * np.log(sum1 / sum2)

        return ans

    if order == 0:
        g = {'mus': mu_ngram, 'sigmas': sigma_ngram, 'weights': weights_ngram}
        f = {'mus': mu_words, 'sigmas': sigma_words, 'weights': weights_words}
    elif order == 1:
        g = {'mus': mu_words, 'sigmas': sigma_words, 'weights': weights_words}
        f = {'mus': mu_ngram, 'sigmas': sigma_ngram, 'weights': weights_ngram}
    else:
        raise RuntimeError

    # KL(g, f)
    return complex_kl(g, f)


def score_ngrams(dictionary, embed, func_type=0, order=0):
    """
    :param dictionary:
    :param embed:
    :param func_type: int
        Which function to use to calculate KL-divergence.
        If type is 0, then the gaussian KL is used;
        if type is 1, then the variational KL is used.
    :param order: int
        If order is 0, then KL(ngram, token) is calculated;
        if order is 1, then KL(token, ngram) is calculated.
    :return: List of scored ngrams
    """
    res = {}

    if func_type == 0:
        func = calculate_closed_kl
    elif func_type == 1:
        func = calculate_variational_kl
    else:
        raise RuntimeError

    vocab_gauss = Vocabulary(dictionary)

    for ngram in ngrams_from_dict(dictionary):
        res[ngram] = func(ngram, vocab_gauss, embed, order)

    return sorted(res.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    raise RuntimeError
