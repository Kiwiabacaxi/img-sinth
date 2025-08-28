# Example prompt structure

## brachiaria_dense_tufts
'brachiaria_dense_tufts': {
    'positive': (
        "directly overhead view of brachiaria grass field, "
        "thick tropical grass growing in natural clumps, "
        "wide green leaf blades, dense vegetation coverage, "
        "scientific research photography, agricultural study, "
        "natural outdoor lighting, soft daylight, "
        "detailed tropical grass texture, realistic pattern, "
        "brachiaria pasture, cattle grazing land, "
        "complete top-down aerial perspective, "
        "no bare patches, full ground coverage, "
        "brazilian tropical grassland research image"
    ),
    'negative': (
        "ground perspective, eye level view, "
        "horizon line, sky, background elements, "
        "fine grass, thin blades, temperate species, "
        "flowers, white blooms, clover, legumes, "
        "manicured lawn, golf course grass, "
        "dramatic lighting, shadows, contrast, "
        "poor quality, distorted, artificial"
    ),
    'description': "Brachiaria em touceiras vista aérea"
}

# Examples with health parameters

## brachiaria_healthy
### Condição 1 - Pastagem saudável
'brachiaria_healthy_condition': {
    'positive': (
        "overhead view of healthy brachiaria brizantha pasture, "
        "thick bright green grass with vigorous growth, "
        "dense uniform coverage, no bare soil visible, "
        "wide leaf blades, robust plant structure, "
        "natural tufted growth pattern, optimal density, "
        "scientific documentation, agricultural research, "
        "natural daylight, top-down perspective"
    ),
    'negative': (
        "ground level, thin grass, sparse coverage, "
        "yellow patches, bare soil, weeds, invasive plants, "
        "artistic photography, decorative lawn"
    ),
    'description': "Brachiaria condição 1 - saudável"
},

### Condição 2 - Leve declínio
'brachiaria_slight_decline': {
    'positive': (
        "aerial view of brachiaria pasture with minor stress signs, "
        "mostly green grass with some lighter colored patches, "
        "good overall coverage but slightly irregular density, "
        "tropical grass showing early stress indicators, "
        "scientific field documentation, research photography, "
        "natural lighting, overhead perspective"
    ),
    'negative': (
        "ground perspective, severe damage, completely yellow, "
        "ornamental setting, artistic effects"
    ),
    'description': "Brachiaria condição 2 - leve declínio"
},

### Condição 3 - Degradação moderada
'brachiaria_moderate_degradation': {
    'positive': (
        "top-down view of moderately degraded brachiaria pasture, "
        "mixed green and yellowish grass patches, "
        "uneven plant density, some small bare soil areas, "
        "tropical grass showing clear stress signs, "
        "reduced vigor compared to healthy pasture, "
        "agricultural study documentation, field research"
    ),
    'negative': (
        "ground level, completely healthy, lush green, "
        "decorative grass, artistic photography"
    ),
    'description': "Brachiaria condição 3 - degradação moderada"
},

### Condição 4 - Degradação avançada
'brachiaria_advanced_degradation': {
    'positive': (
        "overhead documentation of severely degraded brachiaria, "
        "sparse grass coverage with extensive bare soil patches, "
        "yellowing and browning vegetation, weak plant structure, "
        "significant soil compaction visible, erosion signs, "
        "tropical pasture in advanced deterioration state, "
        "scientific documentation of pasture decline"
    ),
    'negative': (
        "healthy vegetation, dense coverage, bright green, "
        "artistic style, decorative setting"
    ),
    'description': "Brachiaria condição 4 - degradação avançada"
},

### Condição 5 - Pastagem deteriorada
'brachiaria_severely_deteriorated': {
    'positive': (
        "aerial view of severely deteriorated brachiaria pasture, "
        "mostly bare compacted soil with scattered dying grass, "
        "brown and yellow stressed vegetation, soil erosion, "
        "failed tropical grassland, extreme degradation, "
        "invasive weeds present, poor soil structure visible, "
        "agricultural research documentation of pasture failure"
    ),
    'negative': (
        "healthy pasture, green vegetation, good coverage, "
        "ornamental lawn, artistic photography"
    ),
    'description': "Brachiaria condição 5 - severamente deteriorada"
}

## panicum_maximum_healthy

### Condição 1 - Panicum saudável
'panicum_healthy_condition': {
    'positive': (
        "overhead view of healthy panicum maximum pasture, "
        "tall robust tropical grass with wide leaf blades, "
        "dense green coverage, vigorous upright growth, "
        "thick stems visible, large leaf structure, "
        "natural clumping pattern, optimal plant density, "
        "scientific documentation, agricultural research, "
        "natural daylight, top-down perspective"
    ),
    'negative': (
        "ground level, short grass, thin blades, sparse coverage, "
        "yellowing, bare soil, weeds, temperate species"
    ),
    'description': "Panicum condição 1 - saudável"
},

### Condição 2 - Leve declínio
'panicum_slight_decline': {
    'positive': (
        "aerial view of panicum pasture with early stress signs, "
        "mostly green tall grass with some pale patches, "
        "good coverage but irregular growth vigor, "
        "large tropical grass showing initial decline, "
        "scientific field documentation, research photography"
    ),
    'negative': (
        "ground perspective, severe damage, short grass, "
        "ornamental lawn, artistic effects"
    ),
    'description': "Panicum condição 2 - leve declínio"
},

### Condição 3 - Degradação moderada
'panicum_moderate_degradation': {
    'positive': (
        "top-down view of moderately degraded panicum pasture, "
        "mixed green and yellow tall grass areas, "
        "uneven plant height, some bare patches visible, "
        "tropical grass with reduced vigor and density, "
        "signs of overgrazing stress, field research documentation"
    ),
    'negative': (
        "healthy dense coverage, uniform green, "
        "decorative setting, artistic photography"
    ),
    'description': "Panicum condição 3 - degradação moderada"
},

### Condição 4 - Degradação avançada
'panicum_advanced_degradation': {
    'positive': (
        "overhead view of severely degraded panicum pasture, "
        "sparse tall grass with extensive bare soil, "
        "yellowing stems, weak plant structure, "
        "soil compaction and erosion visible, "
        "advanced deterioration of tropical grassland"
    ),
    'negative': (
        "lush vegetation, dense coverage, healthy growth, "
        "ornamental grass, artistic style"
    ),
    'description': "Panicum condição 4 - degradação avançada"
},

### Condição 5 - Pastagem deteriorada
'panicum_severely_deteriorated': {
    'positive': (
        "aerial documentation of failed panicum pasture, "
        "mostly bare compacted soil, scattered dying tall grass, "
        "brown withered stems, extreme soil degradation, "
        "invasive weeds colonizing failed grassland, "
        "complete pasture system collapse"
    ),
    'negative': (
        "healthy pasture, green vegetation, good structure, "
        "decorative lawn, artistic photography"
    ),
    'description': "Panicum condição 5 - severamente deteriorado"
}
