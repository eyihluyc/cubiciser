import numpy as np
import cv2
from imagefarm import DIMENSIONS, random_image_adjustment, polygon_area, random_polygon, random_point, transparent_superimposition, fragment_overlay
from random import randint
import math


class Gene:
    def __init__(self, source_img, image_adjustment, mask_vertices, dst_shift, dst_rotation, dst_scaling, applying_strategy):
        self.source_img = source_img
        self.image_adjustment = image_adjustment
        self.mask_vertices = mask_vertices
        self.dst_shift = dst_shift
        self.dst_scaling = dst_scaling
        self.dst_rotation = dst_rotation
        self.applying_strategy = applying_strategy


class Chromosome:
    def __init__(self, genes):
        self.genes = genes
        self.update()

    def update(self):
        self.genes = sorted(self.genes, key=lambda allele: -polygon_area(allele.mask_vertices)*allele.dst_scaling)


def initialize_population(source_images, n):
    return [random_chromosome(source_images) for _ in range(n)]


def decode(chromosome):
    image = np.zeros((512, 512, 3), dtype='uint8')
    for allele in chromosome.genes:
        mask = np.zeros((512, 512), dtype='uint8')
        cv2.fillPoly(mask, pts=[allele.mask_vertices], color=255)
        temp_img = allele.image_adjustment(allele.source_img)
        fragment = cv2.bitwise_and(temp_img, temp_img, mask=mask)
        shiftMatrix = np.float32([[1, 0, allele.dst_shift[0]], [0, 1, allele.dst_shift[1]]])
        fragment = cv2.warpAffine(fragment, shiftMatrix, DIMENSIONS)
        fragment_center = sum(allele.mask_vertices)/len(allele.mask_vertices)
        rotationMatrix = cv2.getRotationMatrix2D((fragment_center[0], fragment_center[1]), allele.dst_rotation, allele.dst_scaling)
        fragment = cv2.warpAffine(fragment, rotationMatrix, DIMENSIONS)
        image = allele.applying_strategy(image, fragment)
    return image


def random_gene(source_images):
    source_img = source_images[randint(0, len(source_images) - 1)]
    image_adjustment = random_image_adjustment()
    mask_vertices = random_polygon(randint(6, 8))
    dst_shift = random_point([0, 0], 100)
    dst_rotation = randint(-20, 20)
    dst_scaling = randint(50, 130)/100
    # applying_strategy = transparent_superimposition if polygon_area(mask_vertices)*dst_scaling > 100000 else fragment_overlay
    applying_strategy = fragment_overlay
    return Gene(source_img, image_adjustment, mask_vertices, dst_shift, dst_rotation,
                dst_scaling, applying_strategy)


def randomize(allele):
    allele.image_adjustment = apply_with_probability(0.7, random_image_adjustment(), allele.image_adjustment)
    allele.mask_vertices = apply_with_probability(0.7, random_polygon(randint(6, 8)), allele.mask_vertices)
    allele.dst_shift = apply_with_probability(0.7, random_point([0, 0], 100), allele.dst_shift)
    allele.dst_rotation = apply_with_probability(0.7, randint(-20, 20), allele.dst_rotation)
    allele.dst_scaling = apply_with_probability(0.7, randint(50, 130)/100, allele.dst_scaling)
    # allele.applying_strategy = transparent_superimposition if polygon_area(allele.mask_vertices)*allele.dst_scaling > 100000 else fragment_overlay


def genes_in_chromosomes():
    return randint(7, 13)


def random_chromosome(source_images):
    n = genes_in_chromosomes()
    genes = [random_gene(source_images) for _ in range(n)]
    chromosome = Chromosome(genes)
    return chromosome


def fitness_function(chromosome, fitness_dict):
    if fitness_dict.get(chromosome) is not None:
        return fitness_dict[chromosome]
    image = decode(chromosome)

    # number of repeated pixels, weighted by number of repetitions (n times repeated -> weighted by (n-1))
    def repetitions_in_genes():
        def gene2polygon_img(allele):
            polygon_img = np.zeros((512, 512), dtype='uint8')
            cv2.fillPoly(polygon_img, pts=[allele.mask_vertices], color=1)
            return polygon_img
        polygons_shadow = sum(map(gene2polygon_img, chromosome.genes))
        return sum(sum(map(lambda x: x*x, polygons_shadow)))/100000

    #  number of non-zero-pixels
    def non_zero_pixels():
        return cv2.countNonZero(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))/150000

    #  number of pixels in canny except long lines from polygons, multiplied by 255.
    def shapes_measure():
        #  j = randint(0, 1000)
        #  cv2.imshow(f"orig{j}", image)

        canny = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), cv2.BORDER_DEFAULT), 175, 250)
        #  cv2.imshow(f"canny{j}", canny)

        def lines_mask(canny):
            lines = cv2.HoughLines(canny, 1, np.pi / 1800, 120, None, 0, 0)
            mask = np.zeros((512, 512), dtype='uint8')
            if lines is not None:
                for i in range(0, len(lines)):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                    cv2.line(mask, pt1, pt2, 255, 6, cv2.LINE_8)
            return mask

        lines = cv2.bitwise_and(canny, canny, mask=lines_mask(canny))
        #  cv2.imshow(f"lines{j}", lines)

        return np.sum(canny - lines)*1.5/1000000

    # print("shapes ", shapes_measure())
    # print("emptiness ", ban_emptiness())
    # print("repetitions ", repetitions_in_genes())
    fitness_dict[chromosome] = 1.5*shapes_measure() + non_zero_pixels() - repetitions_in_genes()
    return fitness_dict[chromosome]


def select(prob_of_highest, population, n, fitness_dict):
    assert 0 < prob_of_highest < 1 or prob_of_highest and n == 1
    sorted_population = sorted(population, key=lambda chromosome: -fitness_function(chromosome, fitness_dict))
    probabilities = [prob_of_highest * (1 - prob_of_highest) ** i for i in range(len(sorted_population))]
    probabilities[-1] /= prob_of_highest
    return np.random.choice(sorted_population, size=n, replace=False, p=probabilities)


def mutate(chromosome, p):
    def almost_sure_mutation(chromosome):
        gene_locus = randint(0, len(chromosome.genes)-1)
        randomize(chromosome.genes[gene_locus])
        chromosome.update()
        return chromosome

    def skip(chromosome):
        return chromosome

    return apply_with_probability(p, almost_sure_mutation, skip, chromosome)


def apply_with_probability(p, func1, func2, *args):
    if randint(0, 99) < p*100:
        if len(args) == 0:
            return func1
        else:
            return func1(*args)
    else:
        if len(args) == 0:
            return func2
        else:
            return func2(*args)


def pair_as_parents(population, n, prob_of_highest, fitness_dict):
    sorted_population = sorted(population, key=lambda chromosome: - fitness_function(chromosome, fitness_dict))
    probabilities = [prob_of_highest * (1 - prob_of_highest) ** i for i in range(len(sorted_population))]
    probabilities[-1] /= prob_of_highest
    return np.random.choice(sorted_population, size=(n, 2), replace=False, p=probabilities)


def crossover(parents):
    chromosome1 = parents[0]
    chromosome2 = parents[1]
    alleles_pool = [*chromosome1.genes, *chromosome2.genes]
    chosen_alleles = np.random.choice(alleles_pool, size=genes_in_chromosomes(), replace=False)
    child = Chromosome(chosen_alleles)
    return child


def decorated(chromosome):
    res = fragment_overlay(cv2.GaussianBlur(chromosome.genes[0].source_img, (7, 7), cv2.BORDER_DEFAULT), decode(chromosome))
    res = cv2.medianBlur(res, 3)
    return res