#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:45:50 2020

@author: gnanfack
"""

config_params = {
	'dataset_name' : "bank_marketing",
	"seed": 111,
	"converge_before": False,
	"weights" : True,
	}

hyper_params = {
	'type_eta' : "eta_variant",
	"value_eta": 40,
	"tol": 0.001,
	"_lambda": 2.,
	"global_steps": 50,
	"bb_steps": 20,
	"stgma_steps": 100,
	"stgma_lr": 0.01,
	"bb_lr": 0.001,
	"theta": 0.2,
	"nb_MC_samples": 10,
	"n_components": 3,
	"nb_units": [10, 20],
	"m_max_min": 10.
}