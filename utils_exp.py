import json

def get_prompt(prompt):
    if prompt == 'all_artists':
        with open('./data/MACE_erase_art_100.json', 'r') as f:
            all_artists = json.load(f)
        prompt = ', '.join(all_artists['erase'])
        preserved = ' '

    if prompt == 'Kelly McKernan':
        prompt = "Kelly Mckernan"
        preserved = ' '
    
    if prompt == 'Thomas Kinkade':
        prompt = "Thomas Kinkade"
        preserved = ' '
    
    if prompt == 'Ajin Demi Human':
        prompt = "Ajin Demi Human"
        preserved = ' '
        
    if prompt == 'Tyler Edlin':
        prompt = "Tyler Edlin"
        preserved = ' '
    
    if prompt == 'Kilian Eng':
        prompt = "Kilian Eng"
        preserved = ' '

    if prompt == 'Van Gogh':
        prompt = "Van Gogh"
        preserved = ' '

    if prompt == 'i2p':
        prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
        preserved = ' '

    if prompt == 'nudity':
        prompt = "nudity"
        preserved = ' '

    if prompt == 'nudity_with_person':
        prompt = "nudity"
        preserved = 'person'

    if prompt == "artifact":
        prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"
    
    if prompt == 'imagenette':
        prompt = ['Cassette Player', 'Chain Saw', 'Church', 'Gas Pump', 'Tench', 'Garbage Truck', 'English Springer', 'Golf Ball', 'Parachute', 'French Horn']
        prompt = 'Cassette Player, Chain Saw, Church, Gas Pump, Tench, Garbage Truck, English Springer, Golf Ball, Parachute, French Horn'
    
    if prompt == 'cassette_player':
        prompt = 'Cassette Player'
        preserved = ' '

    if prompt == 'garbage_truck':
        prompt = 'Garbage Truck'
        preserved = ' '

    if prompt == 'garbage_truck_with_lexus':
        prompt = 'Garbage Truck'
        preserved = 'lexus'

    if prompt == 'garbage_truck_with_road':
        prompt = 'Garbage Truck'
        preserved = 'road'

    if prompt == 'imagenette_v1_wo':
        prompt = 'Cassette Player, Church, Garbage Truck, Parachute, French Horn'
        preserved = ' '

    if prompt == 'imagenette_v2_wo':
        prompt = 'Chain Saw, Gas Pump, Tench, English Springer, Golf Ball'
        preserved = ' '
    
    if prompt == 'imagenette_v3_wo':
        prompt = 'Cassette Player, Chain Saw, Church, Gas Pump, Tench'
        preserved = ' '
    
    if prompt == 'imagenette_v4_wo':
        prompt = 'Garbage Truck, English Springer, Golf Ball, Parachute, French Horn'
        preserved = ' '

    
    if prompt == 'imagenette_small':
        prompt = 'Cassette Player, Church, Garbage Truck, Parachute, French Horn'
        preserved = 'Chain Saw, Gas Pump, Tench, English Springer, Golf Ball'

    if prompt == 'imagenette_v2':
        prompt = 'Chain Saw, Gas Pump, Tench, English Springer, Golf Ball'
        preserved = 'Cassette Player, Church, Garbage Truck, Parachute, French Horn'
    
    if prompt == 'imagenette_v3':
        prompt = 'Cassette Player, Chain Saw, Church, Gas Pump, Tench'
        preserved = 'Garbage Truck, English Springer, Golf Ball, Parachute, French Horn'
    
    if prompt == 'imagenette_v4':
        prompt = 'Garbage Truck, English Springer, Golf Ball, Parachute, French Horn'
        preserved = 'Cassette Player, Chain Saw, Church, Gas Pump, Tench'
    

    return prompt, preserved