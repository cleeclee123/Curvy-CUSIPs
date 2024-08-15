#!/usr/bin/env python3

# SPIA - Income annuity (SPIA and DIA) price calculator
# Copyright (C) 2018 Gordon Irlam
#
# This program may be licensed by you (at your option) under an Open
# Source, Free for Non-Commercial Use, or Commercial Use License.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is free for non-commercial use: you can use and modify it
# under the terms of the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International Public License
# (https://creativecommons.org/licenses/by-nc-sa/4.0/).
#
# A Commercial Use License is available in exchange for agreed
# remuneration.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from statistics import mean


class MonotoneConvex(object):
    """Hagan, Patrick S., and Graeme West. "Methods for constructing a yield curve." Wilmott Magazine, May (2008): 70-81."""

    def collar(self, a, b, c):

        return max(a, min(b, c))

    def last_index(self, term):

        for i in range(len(self.terms) - 1):
            if term < self.terms[i + 1]:
                return i

        return len(self.terms) - 2

    def __init__(
        self,
        terms,
        spots,
        *,
        min_long_term_forward=float("inf"),
        force_forwards_non_negative=True
    ):
        """Initialize a monotone convex object.

        term is an ascending list of bond terms in years, exluding
        zero.

        spots is the corresponding list of continuously compounding
        annualized spot rates.

        min_long_term_forward is the minimum term to use in computing
        the average forward rate. The average forward rate is used as
        the forward rate for terms beyond on the longest term
        specified. If min_long_term_forward exceeds the longest term
        the forward rate of the longest term is used for the forward
        rate of terms beyond the longest term. Set
        min_long_term_forward to 15 to use the Treasury methodology,
        which is to use for the extrapolated forward rate the average
        observed forward rate beyond 15 years. Setting
        min_long_term_forward to a value less than the longest term
        may introduce a discontinuity in forward rates at the longest
        term which may not be desired.

        force_forwards_non_negative specifies whether forward rates
        are forced to be non-negative. If so the negative derivative
        of the spot rate determined price need not match the forward
        rate.

        """

        self.force_forwards_non_negative = force_forwards_non_negative

        assert len(terms) == len(spots)

        # extend the curve to time 0
        terms = [0] + list(terms)
        spots = [spots[0]] + list(spots)

        # step 1
        fdiscrete = []
        for j in range(len(terms) - 1):
            fdiscrete.append(
                (terms[j + 1] * spots[j + 1] - terms[j] * spots[j])
                / (terms[j + 1] - terms[j])
            )

        # step 2
        f = [None]
        for j in range(len(terms) - 2):
            f.append(
                (terms[j + 1] - terms[j])
                / (terms[j + 1 + 1] - terms[j])
                * fdiscrete[j + 1]
                + (terms[j + 2] - terms[j + 1])
                / (terms[j + 2] - terms[j])
                * fdiscrete[j]
            )

        # step 3
        f[0] = fdiscrete[0] - 0.5 * (f[1] - fdiscrete[0])
        f.append(fdiscrete[-1] - 0.5 * (f[-1] - fdiscrete[-1]))
        if force_forwards_non_negative:
            f[0] = self.collar(0, f[0], 2 * fdiscrete[0])
            f[-1] = self.collar(0, f[-1], 2 * fdiscrete[-1])
            for j in range(len(terms) - 2):
                f[j + 1] = self.collar(
                    0, f[j + 1], 2 * min(fdiscrete[j], fdiscrete[j + 1])
                )

        self.terms = tuple(terms)
        self.spots = tuple(spots)
        self.fdiscrete = tuple(fdiscrete)
        self.f = tuple(f)

        if min_long_term_forward < terms[-1]:
            self.long_term_forward = (
                terms[-1] * self.spot(terms[-1])
                - min_long_term_forward * self.spot(min_long_term_forward)
            ) / (terms[-1] - min_long_term_forward)
        else:
            self.long_term_forward = self.f[-1]

    def forward(self, term):
        """Return the continuously compounding annualized forward rate for
        term, term.

        """

        if term <= 0:
            forward = self.f[0]
        elif term == self.terms[-1]:
            forward = self.f[-1]
        elif term > self.terms[-1]:
            forward = self.long_term_forward
        else:
            i = self.last_index(term)
            # the x in (25)
            x = (term - self.terms[i]) / (self.terms[i + 1] - self.terms[i])
            g0 = self.f[i] - self.fdiscrete[i]
            g1 = self.f[i + 1] - self.fdiscrete[i]
            if x == 0:
                g = g0
            elif x == 1:
                g = g1
            elif (g0 < 0 and -0.5 * g0 <= g1 and g1 <= -2 * g0) or (
                g0 > 0 and -0.5 * g0 >= g1 and g1 >= -2 * g0
            ):
                # zone (i)
                g = g0 * (1 - 4 * x + 3 * x**2) + g1 * (-2 * x + 3 * x**2)
            elif (g0 < 0 and g1 > -2 * g0) or (g0 > 0 and g1 < -2 * g0):
                # zone (ii)
                # (29)
                eta = (g1 + 2 * g0) / (g1 - g0)
                # (28)
                if x <= eta:
                    g = g0
                else:
                    g = g0 + (g1 - g0) * ((x - eta) / (1 - eta)) ** 2
            elif (g0 > 0 and 0 > g1 and g1 > -0.5 * g0) or (
                g0 < 0 and 0 < g1 and g1 < -0.5 * g0
            ):
                # zone (iii)
                # (31)
                eta = 3 * g1 / (g1 - g0)
                # (30)
                if x < eta:
                    g = g1 + (g0 - g1) * ((eta - x) / eta) ** 2
                else:
                    g = g1
            elif g0 == 0 and g1 == 0:
                g = 0
            else:
                # zone (iv)
                # (33)
                eta = g1 / (g1 + g0)
                # (34)
                a = -g0 * g1 / (g0 + g1)
                # (32)
                if x <= eta:
                    g = a + (g0 - a) * ((eta - x) / eta) ** 2
                else:
                    g = a + (g1 - a) * ((eta - x) / (1 - eta)) ** 2
                    # (26)
            forward = g + self.fdiscrete[i]

        if self.force_forwards_non_negative:
            forward = max(0, forward)

        return forward

    def spot(self, term):
        """Return the continuously compounding annualized spot rate for term,
        term.

        """

        if term <= 0:
            return self.f[0]
        elif term > self.terms[-1]:
            return (
                self.terms[-1] * self.spots[-1]
                + (term - self.terms[-1]) * self.long_term_forward
            ) / term
        else:
            i = self.last_index(term)
            l = self.terms[i + 1] - self.terms[i]
            # the x in (25)
            x = (term - self.terms[i]) / l
            g0 = self.f[i] - self.fdiscrete[i]
            g1 = self.f[i + 1] - self.fdiscrete[i]
            if x == 0 or x == 1:
                g = 0
            elif (g0 < 0 and -0.5 * g0 <= g1 and g1 <= -2 * g0) or (
                g0 > 0 and -0.5 * g0 >= g1 and g1 >= -2 * g0
            ):
                # zone [i]
                g = l * (g0 * (x - 2 * x**2 + x**3) + g1 * (-(x**2) + x**3))
            elif (g0 < 0 and g1 > -2 * g0) or (g0 > 0 and g1 < -2 * g0):
                # zone (ii)
                # (29)
                eta = (g1 + 2 * g0) / (g1 - g0)
                # (28)
                if x <= eta:
                    g = g0 * (term - self.terms[i])
                else:
                    g = (
                        g0 * (term - self.terms[i])
                        + (g1 - g0) * (x - eta) ** 3 / (1 - eta) ** 2 / 3 * l
                    )
            elif (g0 > 0 and 0 > g1 and g1 > -0.5 * g0) or (
                g0 < 0 and 0 < g1 and g1 < -0.5 * g0
            ):
                # zone (iii)
                # (31)
                eta = 3 * g1 / (g1 - g0)
                # (30)
                if x < eta:
                    g = l * (
                        g1 * x - 1 / 3 * (g0 - g1) * ((eta - x) ** 3 / eta**2 - eta)
                    )
                else:
                    g = l * (2 / 3 * g1 + 1 / 3 * g0) * eta + g1 * (x - eta) * l
            elif g0 == 0 and g1 == 0:
                g = 0
            else:
                # zone (iv)
                # (33)
                eta = g1 / (g1 + g0)
                # (34)
                a = -g0 * g1 / (g0 + g1)
                # (32)
                if x <= eta:
                    g = l * (a * x - 1 / 3 * (g0 - a) * ((eta - x) ** 3 / eta**2 - eta))
                else:
                    g = l * (2 / 3 * a + 1 / 3 * g0) * eta + l * (
                        a * (x - eta) + (g1 - a) / 3 * (x - eta) ** 3 / (1 - eta) ** 2
                    )
                    # (12)
            return (
                self.terms[i] * self.spots[i]
                + self.fdiscrete[i] * (term - self.terms[i])
                + g
            ) / term
