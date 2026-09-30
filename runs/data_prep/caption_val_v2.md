# Caption validation: caption_val_v2.jsonl (50 ViPE videos)

| stat | Qwen recaption | original HF `long` |
|---|---|---|
| n | 50 | 50 |
| words_mean | 116.420 | 122.660 |
| words_median | 116.500 | 123.500 |
| words_p5 | 89.700 | 95.650 |
| words_p95 | 142.650 | 146.000 |
| words_min | 72 | 27 |
| words_max | 153 | 153 |
| sentences_mean | 6.220 | 5.700 |
| starts_with_a_an | 0.900 | 0.740 |
| starts_with_the_video_or_in_this | 0.040 | 0.040 |
| has_newline_or_list | 0.000 | 0.000 |
| mentions_camera | 0.340 | 0.500 |
| top_opening_share | 0.220 | 0.040 |
| starts_young_woman_long_dark_hair | 0.220 | 0.000 |

Embedding cosine (all-mpnet-base-v2): matched mean 0.851 (min 0.696), mismatched mean 0.336; retrieval top-1 (Qwen -> original) 0.98

## Examples

**00019757** (153 vs 145 words)

- Qwen: A young boy with shoulder-length brown hair sits at a small wooden table, wearing a dark green long-sleeve shirt and blue jeans. Across from him, a young girl with light blonde hair tied back in a ponytail, dressed in a white t-shirt and striped shorts, also sits on a wooden chair. Both children are eating from black plates with utensils, appearing engaged in their meal. The boy uses a fork, while the girl holds her spoon in both hands. A green glass sits between them on the table. The setting is a cozy kitchen with light-colored walls, a tiled backsplash, and a window letting in natural light. The table is simple and unadorned except for the plates and glass. The scene is bathed in warm sunlight streaming through the window, creating soft shadows and highlighting the children’s faces and expressions. The mood is peaceful and domestic, capturing a quiet moment of shared mealtime.
- orig: Two children, a boy and a girl, are seated across from each other at a rustic wooden dining table in a cozy kitchen. The boy, with shoulder-length brown hair and wearing a dark green long-sleeved shirt and jeans, is smiling and laughing as he eats from his black plate using a fork. The girl, with blonde hair tied into a high ponytail secured with a green hair tie and dressed in a white t-shirt and striped pants, is also eating with a spoon, her expression animated as she enjoys her meal. Between them on the table sits a translucent green glass. The kitchen features light-colored cabinets and a patchwork tile backsplash, with sunlight streaming through a window on the right, illuminating the scene. Both children are engaged in conversation, their eyes locked on each other, sharing laughter and the simple joy of a shared meal.

**00021511** (121 vs 117 words)

- Qwen: A young woman with long dark hair tied back into a ponytail is seated on a yoga mat in a gym, wearing a black sports bra and black leggings. She holds a small container of what appears to be protein powder or a similar supplement in her right hand while her left hand rests on her bent knee. She brings the container to her mouth and takes a sip, then sets it down on the mat. The gym around her is well-lit, with various pieces of exercise equipment like resistance bands and kettlebells visible in the background. The walls are adorned with motivational posters, and the floor is a light gray color. The setting suggests a focus on fitness and health.
- orig: A fit woman with defined muscles sits cross-legged on a black exercise mat in a well-equipped gym, dressed in a black sports bra and leggings. She holds a small blue packet in her right hand, bringing it to her lips and taking a bite, her eyes closing briefly as she savors the flavor. Her left hand rests gently on her thigh, while her head tilts slightly as she chews. The gym background features hanging resistance bands, kettlebells, and gymnastic rings, with bright lighting illuminating the space. Her expression shifts from focused to satisfied as she enjoys her snack, occasionally glancing around the room with a smile. The atmosphere conveys a sense of energy and dedication to fitness.

**00002010** (129 vs 142 words)

- Qwen: A man stands confidently in a grand hall adorned with intricate carvings and warm, golden light filtering through ornate windows. He wears an elaborate outfit featuring a rich brown tunic and gold armor with decorative patterns, complemented by a flowing red cape draped over his shoulders. His hands are clasped together in front of him, and he appears to be addressing an unseen audience. The setting is majestic, with tall columns and a large, circular ceiling fixture that adds to the opulence. Flanking the room are several braziers emitting flickering flames, casting a warm glow throughout the space. The man’s posture and the detailed surroundings suggest a scene of authority and importance, possibly a royal court or a ceremonial gathering. The overall atmosphere is one of grandeur and solemnity.
- orig: A regal figure stands at the center of an opulent, ancient hall, bathed in dramatic beams of light piercing through an ornate, circular ceiling design. Dressed in an intricately detailed bronze and brown ceremonial armor, complete with shoulder guards, arm bracers, and a flowing cape, the man exudes authority and solemnity. His hands are clasped in front of him, and his gaze is fixed forward, suggesting he is addressing an unseen audience or preparing for a momentous decision. The hall is flanked by towering pillars adorned with golden embellishments, and numerous braziers with flickering flames cast a warm, golden glow across the room. Smoke drifts gently in the air, adding to the solemn and majestic atmosphere. The setting evokes a sense of power, tradition, and divine presence, as if the man is a ruler or a high priest in a sacred ritual.

**00013240** (125 vs 137 words)

- Qwen: A young woman with long dark hair styled in loose curls wears a black dress with sheer sleeves and a deep neckline. Her makeup is dramatic, featuring bold red lipstick and smoky eyeshadow. She approaches another woman who is seated on a bright red leather couch. The seated woman, dressed in a white wedding gown adorned with intricate jewelry, including a necklace and earrings, has her hair styled in an elegant updo with a veil. The standing woman carefully adjusts something on the seated woman's neck, possibly a necklace or a piece of jewelry. The room is warmly lit, casting soft shadows and highlighting the rich colors of the scene. The atmosphere is intimate and focused, suggesting a moment of preparation for a significant event.
- orig: A bride, adorned in a sparkling white wedding gown, sits gracefully on a rich red leather couch, her hair elegantly styled in an updo with a delicate floral headpiece and veil. She wears a dazzling diamond necklace and matching earrings, her nails painted a soft white. Her hands are clasped together, and she gazes downward with a gentle smile, suggesting a moment of quiet reflection or anticipation. As the camera slowly pans to the right, another woman in a black dress with sheer sleeves and a bow tie enters the frame, leaning in close to the bride. She has long, wavy hair and bold makeup, including dark eyeliner and red lipstick. She gently places her hand over the bride’s, offering comfort and support, while both women share a tender, intimate moment, their expressions conveying warmth and affection.

**00026165** (122 vs 123 words)

- Qwen: A bearded man wearing glasses, a light gray long-sleeve shirt, blue shorts, and black gloves rides a light blue road bike along a winding mountain road. He carries a bright yellow pannier bag on the bike. The road curves gently, bordered by a stone barrier on the right and a steep, rocky cliff on the left. The sun shines brightly in the clear blue sky, casting a warm glow over the scene. As he pedals, the camera follows him from behind, capturing the dynamic movement of his bike and the picturesque mountainous landscape around him. The overall feeling is one of adventure and freedom, with the vast expanse of the mountains and the bright sunlight creating a sense of openness and tranquility.
- orig: A bearded man with glasses is cycling along a winding mountain road under a bright, clear sky. He wears a gray long-sleeve shirt and blue shorts, gripping the handlebars of a light blue bicycle that has a yellow bag strapped across the front. The sun shines intensely from behind a towering rocky cliff, casting radiant beams and lens flares across the scene. To his right, a stone barrier lines the road, while the left side features a steep, rocky slope with sparse vegetation. In the distance, the valley unfolds with rolling hills and more mountain peaks under a vast blue sky. The man pedals steadily, maintaining a steady gaze forward, embodying the spirit of adventure and endurance as he navigates the scenic route.

