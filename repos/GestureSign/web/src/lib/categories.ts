export interface SignCategory {
  id: string;
  name: string;
  description: string;
  iconName: string;
  signs: string[];
}

export const SIGN_CATEGORIES: SignCategory[] = [
  {
    id: 'essentials',
    name: 'Greetings & Essentials',
    description: 'Foundational conversational signs: greetings, courtesy, and daily interactions.',
    iconName: 'Sparkles',
    signs: [
      'hello', 'thankyou', 'please', 'bye', 'yes', 'no', 'fine', 'happy', 'mad',
      'help', 'like', 'weus', 'that', 'there', 'who', 'what', 'where', 'why', 'now', 'finish'
    ]
  },
  {
    id: 'animals',
    name: 'Animals & Wildlife',
    description: 'Animals from home, farm, and the wild with vivid visual gestures.',
    iconName: 'Rabbit',
    signs: [
      'alligator', 'animal', 'bee', 'bird', 'bug', 'cat', 'cow', 'dog', 'donkey',
      'duck', 'elephant', 'fish', 'frog', 'giraffe', 'goose', 'hen', 'horse',
      'lion', 'monkey', 'mouse', 'owl', 'pig', 'puppy', 'rooster', 'snake',
      'tiger', 'wolf', 'zebra'
    ]
  },
  {
    id: 'food',
    name: 'Food & Nutrition',
    description: 'Everyday meals, snacks, fruits, and beverages.',
    iconName: 'Utensils',
    signs: [
      'apple', 'carrot', 'cereal', 'chocolate', 'food', 'gum', 'icecream',
      'milk', 'nut', 'pizza', 'snack', 'strawberry', 'taste', 'thirsty', 'water'
    ]
  },
  {
    id: 'objects',
    name: 'Everyday Objects',
    description: 'Common household items, technology, tools, and transportation.',
    iconName: 'Box',
    signs: [
      'airplane', 'balloon', 'bed', 'boat', 'book', 'car', 'chair', 'closet',
      'helicopter', 'jacket', 'jeans', 'lamp', 'mirror', 'napkin', 'pan',
      'pencil', 'penny', 'radio', 'refrigerator', 'scissors', 'shirt', 'shoe',
      'sink', 'soap', 'socks', 'table', 'toothbrush', 'toy', 'TV', 'vacuum', 'zipper'
    ]
  },
  {
    id: 'family',
    name: 'People & Family',
    description: 'Relationships, kinship, and person descriptors.',
    iconName: 'Users',
    signs: [
      'aunt', 'boy', 'brother', 'child', 'dad', 'girl', 'grandfather',
      'grandmother', 'hesheit', 'man', 'person', 'sister', 'uncle', 'yourself'
    ]
  },
  {
    id: 'colors',
    name: 'Colors',
    description: 'Primary, secondary, and neutral colors in ASL.',
    iconName: 'Palette',
    signs: [
      'black', 'blue', 'brown', 'green', 'red', 'white', 'yellow'
    ]
  },
  {
    id: 'body',
    name: 'Body & Sensations',
    description: 'Body parts, facial landmarks, and health sensations.',
    iconName: 'HeartPulse',
    signs: [
      'arm', 'cheek', 'chin', 'ear', 'eye', 'face', 'feet', 'finger',
      'foot', 'hair', 'head', 'hear', 'lips', 'mouth', 'nose', 'sick',
      'tongue', 'tooth'
    ]
  },
  {
    id: 'actions',
    name: 'Actions & Verbs',
    description: 'Common motions, verbs, and activities.',
    iconName: 'Activity',
    signs: [
      'blow', 'callonphone', 'clean', 'close', 'cry', 'dance', 'fall',
      'find', 'give', 'go', 'goahead', 'hide', 'jump', 'kiss', 'listen',
      'look', 'make', 'pretend', 'read', 'ride', 'rip', 'run', 'say',
      'see', 'shhh', 'sit', 'sleep', 'smile', 'stuck', 'talk', 'touch',
      'wait', 'wake'
    ]
  }
];

export function getCategoryForSign(signName: string): SignCategory | undefined {
  const norm = signName.toLowerCase().replace(/[^a-z]/g, '');
  return SIGN_CATEGORIES.find((cat) =>
    cat.signs.some((s) => s.toLowerCase().replace(/[^a-z]/g, '') === norm)
  );
}

