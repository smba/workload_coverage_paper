#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:34:40 2022

@author: stefan
"""

import pandas as pd
from pycosa.util import reconstruct_categorical_variable

systems = [
    'jump3r',
    'jadx',
    'batik',
    'kanzi',
    'h2',
    'dconvert',    
]

for sys in systems:
    df = pd.read_csv('data/{}_measurements.csv'.format(sys))
    
    if sys == 'jump3r':
        df = reconstruct_categorical_variable(
            df,
            {'Resampling_8': 8,
             'Resampling_22': 22,
             'Resampling_44': 44,
             'Resampling_48': 48},
            'Resampling'
            
        )
        
        df = reconstruct_categorical_variable(
            df,
            {'ABR_320': 320,
             'ABR_160': 160,
             'ABR_8': 8},
            'ABR'
        )
        df = reconstruct_categorical_variable(
            df,
            {'CBR_8': 8, 
             'CBR_160': 160,
             'CBR_320': 320},
            'CBR'
        )
        
        df = reconstruct_categorical_variable(
            df,
            {'Lowpass_5000': 5000,
             'Lowpass_10000': 10000,
             'Lowpass_20000': 20000,
             },
            'Lowpass'
        )
        df = reconstruct_categorical_variable(
            df,
            {'Highpass_20000': 20000,
             'Highpass_25000': 25000,
             'Highpass_30000': 30000,},
            'Highpass'
        )
        
        df = reconstruct_categorical_variable(
            df,
            {'LowpassWidth_1000': 1000,
             'LowpassWidth_2000': 2000,
             },
            'LowpassWidth'
        )
        df = reconstruct_categorical_variable(
            df,
            {'HighpassWidth_1000': 1000,
             'HighpassWidth_2000': 2000,},
            'HighpassWidth'
        )

        df = reconstruct_categorical_variable(
            df,
            {'VBRRange_low': 0,
             'VBRRange_wide': 1,
             'VBRRange_high': 2},
            'VBRRange'
        )

    if sys == 'batik':
        df = reconstruct_categorical_variable(
            df,
            {
                'dpi50': 50,
                'dpi100': 100,
                'dpi200': 200,
                'dpi400': 400
            },
            'dpi'
        )
        df = reconstruct_categorical_variable(
            df,
            {
                'sd_resolution': 307200,
                'full_hd_resolution': 2073600,
                '4k_resolution': 8294400
            },
            'resolution'
        )
        df = reconstruct_categorical_variable(
            df,
            {
                'low_quality': 0.1,
                'high_quality': 0.9,
            },
            'quality'
        )
        df = reconstruct_categorical_variable(
            df,
            {
                'indexed_1': 1,
                'indexed_2': 2,
                'indexed_4': 4,
                'indexed_8': 8
            },
            'indexed'
        )
        
    if sys == 'kanzi':
        df = reconstruct_categorical_variable(
            df,
            {
                'jobs_1': 1,
                'jobs_4': 4,
                'jobs_8': 8
            },
            'jobs'
        )
        df = reconstruct_categorical_variable(
            df,
            {
                'blocksize_1KB': 1000,
                'blocksize_1MB': 1000000,
                'blocksize_1GB': 1000000000,
            },
            'blocksize'
        )
        
    if sys == 'jadx':
        df = reconstruct_categorical_variable(
            df,
            {
                'single_threaded': 1,
                'multi_threaded': 8,
            },
            'thread-count'
        )
        
        
    if sys == 'dconvert':
        df = reconstruct_categorical_variable(
            df,
            {
                'single_threaded': 1,
                'multi_threaded': 8,
            },
            'threads'
        )
        
        df = reconstruct_categorical_variable(
            df,
            {
                'low_compression': 0.9,
                'high_compression': 0.1,
            },
            'compression'
        )
        
        df = reconstruct_categorical_variable(
            df,
            {
                'scale10': 10,
                'scale90': 90,
            },
            'threads'
        )
    
    if sys == 'h2':
        df = reconstruct_categorical_variable(
            df,
            {
                'MV_STORE': 1,
                'PAGE_STORE': 0,
            },
            'MVSTORE'
        )
    df.to_csv('datax/{}_measurements.csv'.format(sys), index=False)
        
    
        