#!/usr/bin/env python

from pollos_petrel import LogisticModel, LinearModel

linear = LinearModel()
linear.write_submission()

logistic = LogisticModel()
logistic.write_submission()
