from faker import Faker 
import random
import json


dataset = []


dataset.append(["name","citizenship","age","max_education","race","gender","credit_score"])

class CSFaker:
    citizenships = ['ar_EG', 'ar_PS', 'ar_SA', 'bg_BG', 'cs_CZ', 'de_DE', 'dk_DK', 'el_GR', 'en_AU', 'en_CA', 'en_GB', 'en_US', 'es_ES', 'es_MX', 'et_EE', 'fa_IR', 'fi_FI', 'fr_FR', 'hi_IN', 'hr_HR', 'hu_HU', 'it_IT', 'ja_JP', 'ko_KR', 'lt_LT', 'lv_LV', 'ne_NP', 'nl_NL', 'no_NO', 'pl_PL', 'pt_BR', 'pt_PT', 'ro_RO', 'ru_RU', 'sl_SI', 'sv_SE', 'tr_TR', 'uk_UA', 'zh_CN', 'zh_TW', 'ka_GE']
    def person(self):
        c = random.choice(self.citizenships)
        f = Faker(c)
        return {"name": f.name(), "citizenship": c}
    
    def generator(self):
        #Name
        name = self.person()['name']
        #Citizenship
        citizenship = self.person()['citizenship']
        #Age
        age = random.randint(18, 100)
        #Education
        educ = ['High School', 'college', 'masters', 'phd', 'genius']
        education = random.choice(educ)
        #Race
        race = ['almond', 'oreo', 'quaker']
        race = random.choice(race)
        #Gender
        gender = random.randint(0, 1)
        #CreditScore
        base = 100

        if(education == 'High School'):
            base *= 0.5
        elif (education == 'college'):
            base *= 1
        elif(education == 'masters'):
            base *= 2
        elif(education == 'phd'):
            base *= 3
        else:
            base *= 4
        
        if(race == 'almond'):
            base*= 0.5
        elif (race == 'quaker'):
            base *= 1.5
        else:
            pass
        
        if(gender == 1):
            base *= 1.25

        base *= (age/100.0)
        credit_score = base

        return {"name": name, "citizenship": citizenship, 'age': age, 'education': education, 'race': race, 'gender': gender, 'credit_score': credit_score}

dataset = []
    
for i in range(1000):
    c = CSFaker()
    dataset.append(c.generator())

with open('CreditScoreData.json', 'w') as outfile:
    json.dump(dataset, outfile)

