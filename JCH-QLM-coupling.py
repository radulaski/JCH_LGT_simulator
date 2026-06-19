"""
Example use:

(with lambda):
    python JCH-QLM-coupling.py -D1 100 -g 300 -J 0.9 -l 36

(with deltas):
    python JCH-QLM-coupling.py -D1 100 -g 300 -J 0.9 -D2 160.33626918698374 -D3 35.24213732330507

"""


import numpy as np
import argparse
from scipy.optimize import fsolve

def get_coupling(D1, D2, D3, g, J):
    """ Returns the coupling t given the set of JCH parameters. """
    return J**2 * (0.25 * (2 ** -0.5) * (np.sqrt(1 + D1 * ((D1 ** 2 + 4 * (g ** 2)) ** \
-0.5))) * (np.sqrt(1 -(D2 * ((D2 ** 2 + 4 * (g ** 2)) ** -0.5)))) * \
(np.sqrt(1 + D3 * ((D3 ** 2 + 4 * (g ** 2)) ** -0.5))) * (-0.5 * \
(np.sqrt(1 + D2 * ((D2 ** 2 + 4 * (g ** 2)) ** -0.5))) * (np.sqrt(1 \
-(D2 * ((D2 ** 2 + 8 * (g ** 2)) ** -0.5)))) + (2 ** -0.5) * \
(np.sqrt(1 -(D2 * ((D2 ** 2 + 4 * (g ** 2)) ** -0.5)))) * (np.sqrt(1 \
+ D2 * ((D2 ** 2 + 8 * (g ** 2)) ** -0.5)))) * (-2 / (-D2 + D3 + \
np.sqrt(D2 ** 2 + 4 * (g ** 2)) + np.sqrt(D3 ** 2 + 4 * (g ** 2))) + \
2 / (D1 -2 * D2 + np.sqrt(D1 ** 2 + 4 * (g ** 2)) -(np.sqrt(D2 ** 2 + \
4 * (g ** 2))) -(np.sqrt(D2 ** 2 + 8 * (g ** 2))))) + 0.25 * (2 ** \
-0.5) * (np.sqrt(1 + D1 * ((D1 ** 2 + 4 * (g ** 2)) ** -0.5))) * \
(np.sqrt(1 + D2 * ((D2 ** 2 + 4 * (g ** 2)) ** -0.5))) * (np.sqrt(1 + \
D3 * ((D3 ** 2 + 4 * (g ** 2)) ** -0.5))) * (0.5 * (np.sqrt(1 -(D2 * \
((D2 ** 2 + 4 * (g ** 2)) ** -0.5)))) * (np.sqrt(1 -(D2 * ((D2 ** 2 + \
8 * (g ** 2)) ** -0.5)))) + (2 ** -0.5) * (np.sqrt(1 + D2 * ((D2 ** 2 \
+ 4 * (g ** 2)) ** -0.5))) * (np.sqrt(1 + D2 * ((D2 ** 2 + 8 * (g ** \
2)) ** -0.5)))) * (-2 / (-D2 + D3 -(np.sqrt(D2 ** 2 + 4 * (g ** 2))) \
+ np.sqrt(D3 ** 2 + 4 * (g ** 2))) + 2 / (D1 -2 * D2 + np.sqrt(D1 ** \
2 + 4 * (g ** 2)) + np.sqrt(D2 ** 2 + 4 * (g ** 2)) -(np.sqrt(D2 ** 2 \
+ 8 * (g ** 2))))) + 0.25 * (2 ** -0.5) * (np.sqrt(1 + D1 * ((D1 ** 2 \
+ 4 * (g ** 2)) ** -0.5))) * (np.sqrt(1 -(D2 * ((D2 ** 2 + 4 * (g ** \
2)) ** -0.5)))) * (np.sqrt(1 + D3 * ((D3 ** 2 + 4 * (g ** 2)) ** \
-0.5))) * (-0.5 * (np.sqrt(1 + D2 * ((D2 ** 2 + 4 * (g ** 2)) ** \
-0.5))) * (np.sqrt(1 -(D2 * ((D2 ** 2 + 8 * (g ** 2)) ** -0.5)))) + \
(2 ** -0.5) * (np.sqrt(1 -(D2 * ((D2 ** 2 + 4 * (g ** 2)) ** -0.5)))) \
* (np.sqrt(1 + D2 * ((D2 ** 2 + 8 * (g ** 2)) ** -0.5)))) * (-2 / (D1 \
-D2 + np.sqrt(D1 ** 2 + 4 * (g ** 2)) + np.sqrt(D2 ** 2 + 4 * (g ** \
2))) + 2 / (-2 * D2 + D3 -(np.sqrt(D2 ** 2 + 4 * (g ** 2))) + \
np.sqrt(D3 ** 2 + 4 * (g ** 2)) -(np.sqrt(D2 ** 2 + 8 * (g ** 2))))) \
+ 0.25 * (2 ** -0.5) * (np.sqrt(1 + D1 * ((D1 ** 2 + 4 * (g ** 2)) ** \
-0.5))) * (np.sqrt(1 + D2 * ((D2 ** 2 + 4 * (g ** 2)) ** -0.5))) * \
(np.sqrt(1 + D3 * ((D3 ** 2 + 4 * (g ** 2)) ** -0.5))) * (0.5 * \
(np.sqrt(1 -(D2 * ((D2 ** 2 + 4 * (g ** 2)) ** -0.5)))) * (np.sqrt(1 \
-(D2 * ((D2 ** 2 + 8 * (g ** 2)) ** -0.5)))) + (2 ** -0.5) * \
(np.sqrt(1 + D2 * ((D2 ** 2 + 4 * (g ** 2)) ** -0.5))) * (np.sqrt(1 + \
D2 * ((D2 ** 2 + 8 * (g ** 2)) ** -0.5)))) * (-2 / (D1 -D2 + \
np.sqrt(D1 ** 2 + 4 * (g ** 2)) -(np.sqrt(D2 ** 2 + 4 * (g ** 2)))) + \
2 / (-2 * D2 + D3 + np.sqrt(D2 ** 2 + 4 * (g ** 2)) + np.sqrt(D3 ** 2 \
+ 4 * (g ** 2)) -(np.sqrt(D2 ** 2 + 8 * (g ** 2))))))


def erg(pm, n, omega, D, g):

    """ Return energy of n^pm polarion. """
    assert pm in [+1, -1]
    if n == 0:
        return 0
    else:
        return (omega -D) * n + D/2 + pm * np.sqrt((D/2)**2 + g**2 * n)

def get_D2D3(D1, lambd, g):
    """ Returns D2 and D3 satisfying energy-level matching given D1, lambda, g. """
    omega=1 # omega drops out so we can just set it to zero

    eq1 = lambda d3: erg(-1, 1, omega, d3, g) - (erg(-1, 1, omega, D1, g) + lambd)
    D3 = fsolve(eq1, D1)[0]

    eq2tmp = lambda d2, d3: erg(-1, 1, omega, d3, g) + erg(-1, 1, omega, D1, g)  - erg(-1, 2, omega, d2, g) 
    eq2 = lambda d2: eq2tmp(d2, D3)

    D2 = fsolve(eq2, (D1+D3)/2)[0]

    return D2, D3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-D1", "--delta1",
        nargs=1,
        type=float,
        required=True,
        metavar=("D1"),
        help="JCH parameter D1 (delta_1)"
    )

    parser.add_argument(
        "-g", "--g",
        nargs=1,
        type=float,
        required=True,
        metavar=("g"),
        help="JCH parameter g (stong coupling)"
    )

    parser.add_argument(
        "-J", "--J",
        nargs=1,
        type=float,
        required=True,
        metavar=("J"),
        help="JCH parameter J (weak coupling)"
    )

    parser.add_argument(
        "-D2", "--delta2",
        nargs=1,
        type=float,
        required=False,
        metavar=("D2"),
        help="JCH parameter D2 (delta_3)"
    )

    parser.add_argument(
        "-D3", "--delta3",
        nargs=1,
        type=float,
        required=False,
        metavar=("D3"),
        help="JCH parameter D3 (delta_3)"
    )

    parser.add_argument(
        "-l", "--lambd",
        nargs=1,
        type=float,
        required=False,
        metavar=("lambd"),
        help="Parameter lambda for energy-level matching"
    )


    args = parser.parse_args()

    if not args.lambd and (not args.delta2 or not args.delta3):
        raise TypeError("You must specify either specify all delta parameters of delta_1 and lambda.")

    elif (args.delta2 or args.delta3) and args.lambd:
        print("WARNING: Both lambda and deltas were specified. The program will ignore lambda and use the specified deltas.")

    
    if args.delta2 and args.delta3:
        D2, D3 = args.delta2[0], args.delta3[0]
    else:
        D2, D3 = get_D2D3(args.delta1[0], args.lambd[0], args.g[0])


    t = get_coupling(args.delta1[0], D2, D3, args.g[0], args.J[0])

    print(f"Deltas:   D1 = {args.delta1[0]}, D2 = {D2}, D3 = {D3}")
    print(f"Coupling: t = {t}, t/J = {t/args.J[0]}")

if __name__ == "__main__":
    main()