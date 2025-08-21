binary_cols = [
    'GENDER',
    'ASA',
    'DM',
    'PRERT',
    'PRECT',
    'PRIOREX',
    'SP',
    'RECUR',
    'T',
    'N',
    'DEFECT_TYPE',
    'FLAP',
    'BT',
    'REOPEN',
    'WOUNDINF',
    'EXPOSURE',
    'MEDEXPOSURE',
    'POSTRT',
    'POSTCT'
]

numerical_cols = [
    'AGE',
    'BMI',
    'OPTIME',
    'LENGTH',
    'ISCHEMICTIME',
    'ADMISSION',
    'HGB',
    'ALB',
    'EXPOSUREFU'
]

nominal_cols = [
    'SITE', 
    'JEWER', 
    'PLATE', 
    'TXEXPOSURE'
]

ordinal_cols = [
    'STAGE', 
    'OSTEOTOMY'
]

def get_feature_lists():
    return {
        "binary_cols": list(binary_cols),
        "numerical_cols": list(numerical_cols),
        "nominal_cols": list(nominal_cols),
        "ordinal_cols": list(ordinal_cols),
    }