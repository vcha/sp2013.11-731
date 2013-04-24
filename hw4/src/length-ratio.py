import sys
import numpy

def main():
    stats = []
    for line in sys.stdin:
        source, target, _ = line.decode('utf8').split(' ||| ')
        stats.append(len(target)/float(len(source)))
    print 'avg =', numpy.average(stats)
    print 'std =', numpy.std(stats)

if __name__ == '__main__':
    main()

