def workclass(wclass):
    levels = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
              'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?']
    return levels.index(wclass)


def education(edu):
    levels = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
              'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters',
              '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
    return levels.index(edu)

def marital_status(ms):
    levels = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
              'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
    return levels.index(ms)


def occupation(occ):
    levels = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
              'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
              'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
              'Transport-moving', 'Priv-house-serv', 'Protective-serv',
              'Armed-Forces', '?']
    return levels.index(occ)


def relationship(rsh):
    levels = ['Wife', 'Own-child', 'Husband', 'Not-in-family','Other-relative',
              'Unmarried']
    return levels.index(rsh)


def race(rac):
    levels = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other',
              'Black']
    return levels.index(rac)


def sex(s):
    levels = ['Female', 'Male']
    return levels.index(s)


def native_country(nc):
    levels = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
              'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan',
              'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras',
              'Philippines', 'Italy', 'Poland', 'Jamaica','Vietnam', 'Mexico',
              'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos',
              'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
              'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia','El-Salvador',
              'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?']
    return levels.index(nc)


def accept(acc):
    if '<' in acc:
        return 0
    return 1