import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.special
import numpy.linalg
import time
import datetime
import sys

np.random.seed(0)

class Target:
    def pdf(self, z):
        raise NotImplementedError()

    def logpdf(self, z):
        raise NotImplementedError()

    def likelihood(self, z):
        return self.pdf(z) #times some constant

    def loglikelihood(self, z):
        return self.logpdf(z) #times some constant

    def plot(self, label=None):
        raise NotImplementedError()


class ApproximatingDistribution:
    def varying_theta(self):
        raise NotImplementedError()

    def set_varying_theta(self, varying_theta):
        raise NotImplementedError()

    def pdf(self, z):
        raise NotImplementedError()

    def logpdf(self, z):
        raise NotImplementedError()

    def quasi_rvs(self, size):
        raise NotImplementedError()

    def plot(self, label=None):
        raise NotImplementedError()

class NormalDistribution(ApproximatingDistribution):
    def __init__(self, theta):
        self.theta = theta

    def varying_theta(self):
        return self.theta

    def set_varying_theta(self, varying_theta):
        self.theta = varying_theta

    def parms(self):
        return {
            'loc': self.theta[0],
            'scale': np.exp(self.theta[1])
        }

    def pdf(self, z):
        return scipy.stats.norm.pdf(z, **self.parms())

    def logpdf(self, z):
        return scipy.stats.norm.logpdf(z, **self.parms())

    def quasi_rvs(self, size):
        return scipy.stats.norm.ppf(np.linspace(start=1/size/2, stop=1-1/size/2, num=size), **self.parms())

    def plot(self, label=None):
        r = 4*np.exp(self.theta[1])
        z = np.linspace(start=self.theta[0]-r, stop=self.theta[0]+r, num=100)
        plt.plot(z, self.pdf(z), label=label)

#Mixture did not work at all? Try more?
class BiMixtureDistribution(ApproximatingDistribution):
    def __init__(self, fixed_distribution, varying_distribution, probability):
        self.fixed_distribution = fixed_distribution
        self.varying_distribution = varying_distribution
        self.logit_probability = scipy.special.logit(probability)

    def varying_theta(self):
        return np.append(self.varying_distribution.varying_theta(), self.logit_probability)

    def set_varying_theta(self, varying_theta):
        self.varying_distribution.set_varying_theta(varying_theta[:-1])
        self.logit_probability = varying_theta[-1]

    def pdf(self, z): #expected pdf - hope this works. otherwise redo
        probability = scipy.special.expit(self.logit_probability)
        return (1-probability)*self.fixed_distribution.pdf(z) + probability*self.varying_distribution.pdf(z)

    def logpdf(self, z):
        return np.log(self.pdf(z))

    def plot(self, label=None):
        z = np.linspace(start=-100, stop=100, num=1000)
        plt.plot(z, self.pdf(z), label=label)

#http://dustintran.com/papers/TranRanganathBlei2016.pdf
class GaussianProcess:
    pass

def maximum_likelihood(target, theta0):
    return scipy.optimize.minimize(lambda theta: -target.loglikelihood(theta), x0=theta0)

def laplace_approximation(target, theta0):
    result = maximum_likelihood(target, theta0)
    return NormalDistribution(theta=[result.x, np.log(np.sqrt(result.hess_inv[0][0]))])

def forward_KL(target, approximation): #Forward Kullback-Leibler divergence
    def point_lower_bound(z):
        return approximation.logpdf(z) - target.loglikelihood(z)

    def integrand(z):
        return target.pdf(z)*point_lower_bound(z)

    def lower_bound(varying_theta):
        approximation.set_varying_theta(varying_theta)
        quad_result = scipy.integrate.quadrature(integrand, a=-50, b=50, maxiter=500) #this is unrealistic
        return quad_result[0]

    result = scipy.optimize.minimize(fun=lambda varying_theta: -lower_bound(varying_theta), x0=approximation.varying_theta())
    approximation.set_varying_theta(result.x)
    print('forward_KL result')
    print(result)
    return approximation

def backward_KL(target, approximation): #Backward Kullback-Leibler divergence
    def point_lower_bound(z):
        return target.loglikelihood(z) - approximation.logpdf(z)

    def integrand(z):
        return approximation.pdf(z)*point_lower_bound(z)

    def lower_bound(varying_theta):
        approximation.set_varying_theta(varying_theta)
        quad_result = scipy.integrate.quadrature(integrand, a=-100, b=100, maxiter=500) #this is unrealistic
        return quad_result[0]

    result = scipy.optimize.minimize(fun=lambda varying_theta: -lower_bound(varying_theta), x0=approximation.varying_theta())
    approximation.set_varying_theta(result.x)
    print('backward_KL result')
    print(result)
    return approximation

def forward_chi2(target, approximation): #Forward Chi-Squared divergence https://arxiv.org/pdf/1611.00328.pdf
    def point_upper_bound(z):
        return np.square(target.likelihood(z)/approximation.pdf(z))/2 #can change to another power chi-n

    def integrand(z):
        return approximation.pdf(z)*point_upper_bound()

    def upper_bound(varying_theta):
        approximation.set_varying_theta(varying_theta)
        # quad_result = scipy.integrate.quadrature(integrand, a=-100, b=100, maxiter=500) #this is unrealistic
        # return quad_result[0]
        return np.average(point_upper_bound(approximation.quasi_rvs(100)))


    result = scipy.optimize.minimize(fun=upper_bound, x0=approximation.varying_theta())
    approximation.set_varying_theta(result.x)
    print('forward_chi2 result')
    print(result)
    return approximation

def backward_total_variation(target, approximation): #Backward total variation distance divergence https://en.wikipedia.org/wiki/F-divergence
    pass

def quasi_independence_metropolis(target, approximation, size):

    samples = approximation.quasi_rvs(size)

    def alpha(old_z, new_z): #acceptance probability
        return min(1, target.likelihood(new_z)*approximation.pdf(old_z)/target.likelihood(old_z)/approximation.pdf(new_z))

    def transition_probability(from_index, to_index):
        if from_index == to_index:
            return 1 - np.sum([transition_probability(from_index, to_index) for to_index in range(size) if from_index != to_index])
        return alpha(samples[from_index], samples[to_index]) * 1/size

    def transition_matrix(): #(transposed)
        return np.matrix([[transition_probability(from_index, to_index) for from_index in range(size)] for to_index in range(size)])

    result = scipy.linalg.lstsq(a=np.vstack([transition_matrix()-np.eye(size), np.ones((1,size))]), b=np.vstack([np.zeros((size,1)), 1])) #too expensive with many samples? there are other solutions
    weights = result[0].flatten()

    return (samples, weights)


# def improve_quasi_independence_metropolis(target, approximation, size):
#
#     def upper_bound(varying_theta):
#         approximation.set_varying_theta(varying_theta)
#         (samples, weights) = quasi_independence_metropolis(target, approximation, size)
#         l = target.likelihood(samples)
#         return np.average(weights*np.square(l/sum(l)/weights)/2)
#
#     result = scipy.optimize.minimize(fun=upper_bound, x0=approximation.varying_theta())
#     approximation.set_varying_theta(result.x)
#     (samples, weights) = quasi_independence_metropolis(target, approximation, size)
#     print('improve_quasi_independence_metropolis result')
#     print(result)
#     return (samples, weights, approximation)

class NormalTarget(Target):
    def pdf(self, z):
        return scipy.stats.norm.pdf(z, loc=2, scale=4)

    def logpdf(self, z):
        return scipy.stats.norm.logpdf(z, loc=2, scale=4)

    def plot(self):
        x = np.linspace(start=-10, stop=14, num=100)
        plt.plot(x, self.likelihood(x))

class BivariateNormalTarget(BiMixtureDistribution, Target):
    def __init__(self):
        super().__init__(NormalDistribution([0, np.log(10)]), NormalDistribution([40, np.log(10)]), 0.2)

def approximate_target(target):
    
    mle = maximum_likelihood(target, theta0=0)
    la = laplace_approximation(target, theta0=0)
    fkl = forward_KL(target, approximation=NormalDistribution(theta=[2, 4]))
    kl = backward_KL(target, approximation=NormalDistribution(theta=[2, 4]))
    chi2 = forward_chi2(target, approximation=NormalDistribution(theta=[2, 4]))

    target.plot(label='True distribution')
    plt.axvline(x=mle.x, ls='dashed', c='black', label='Maximum likelihood')
    la.plot(label='Laplace approximation')
    (samples, weights) = quasi_independence_metropolis(target, approximation=la, size=200)
    plt.hist(samples, weights=weights, histtype='step', normed=True, bins=20, label='Quasi Independence Metropolis (from Laplace approximation)')
    plt.legend(loc='upper right')
    plt.show()

    target.plot(label='True distribution')
    kl.plot(label='Backward KL')
    (samples, weights) = quasi_independence_metropolis(target, approximation=kl, size=200)
    plt.hist(samples, weights=weights, histtype='step', normed=True, bins=20, label='Quasi Independence Metropolis (from backward KL)')
    plt.legend(loc='upper right')
    plt.show()

    target.plot(label='True distribution')
    la.plot(label='Laplace approximation')
    fkl.plot(label='Forward KL (unrealistic)')
    kl.plot(label='Backward KL')
    chi2.plot(label='Forward Chi-squared')
    plt.legend(loc='upper right')
    plt.show()

    #Does not work...
    # m2 = backward_KL(target, approximation=BiMixtureDistribution(fixed_distribution=NormalDistribution([0,10]), varying_distribution=NormalDistribution([40,10]), probability=0.3))
    # m2.plot()

    target.plot(label='True distribution')
    chi2.plot(label='Forward Chi-squared')
    (samples, weights) = quasi_independence_metropolis(target, approximation=chi2, size=200)
    plt.hist(samples, weights=weights, histtype='step', normed=True, bins=20, label='Quasi Independence Metropolis (from forward Chi-squared)')
    plt.legend(loc='upper right')
    plt.show()

    # (samples, weights, approximation) = improve_quasi_independence_metropolis(target, approximation=chi2, size=10)
    # approximation.plot()
    # plt.hist(samples, weights=weights, histtype='step', normed=True, bins=20)



#approximate_target(NormalTarget())

approximate_target(BivariateNormalTarget())
