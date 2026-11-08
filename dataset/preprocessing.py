import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

RAW = '/home/eldian/Documents/DA_BugBucket/Dataset/Bug_meta_data_230k_all.csv'
LABELED = '/home/eldian/Documents/DA_BugBucket/bugbucket_labeled.csv'
OUT = '/home/eldian/Documents/DA_BugBucket/bugbucket_clean.csv'

groups = {
    'User Interface': [
        'Address Bar', 'Theme', 'Tabbed Browser', 'Bookmarks & History',
        'Settings UI', 'Menus', 'Toolbars and Customization', 'New Tab Page',
        'Panning and Zooming', 'Mail Window Front End', 'Message Compose Window',
        'DOM: UI Events & Focus Handling', 'Session Restore', 'Widget: Gtk',
        'Widget: Cocoa', 'Widget: Win32', 'Folder and Message Lists',
        'Messaging System', 'DOM: Navigation', 'DOM: Editor', 'Search',
        'Debugger', 'Console', 'Inspector', 'DOM: Core & HTML'],
    'Graphics': [
        'Graphics', 'Graphics: WebRender', 'Graphics: CanvasWebGL',
        'Graphics: Canvas2D', 'SVG', 'Graphics: WebGPU', 'Graphics: ImageLib',
        'Graphics: Text', 'Graphics: Color Management', 'Graphics: Layers',
        'Graphics: Image Blocking', 'Layout', 'CSS Parsing and Computation',
        'Layout: Text and Fonts', 'Layout: Images, Video, and HTML Frames',
        'PDF Viewer'],
    'Media': [
        'Audio/Video: Playback', 'WebRTC: Audio/Video', 'WebRTC', 'Audio/Video',
        'Web Audio', 'Audio/Video: cubeb', 'Audio/Video: Web Codecs', 'Media',
        'Video/Audio Controls', 'Audio/Video: GMP', 'WebRTC: Signaling',
        'Audio/Video: MediaStreamGraph', 'Audio/Video: Recording'],
    'Security and Privacy': [
        'Security', 'DOM: Security', 'Privacy: Anti-Tracking',
        'Privacy: Site Reports', 'Privacy', 'Security: PSM', 'Security: OpenPGP',
        'Security: Process Sandboxing', 'Security: S/MIME', 'Security: RLBox',
        'Security: CAPS', 'Security: OTR', 'Security: iOS', 'Security: General',
        'Security Block-lists, Allow-lists, and other State',
        'Pontoon Security Issues', 'Bleach-security',
        'Operations: Firefox Private Network'],
    'Networking': [
        'Networking', 'DOM: Networking', 'Netmonitor', 'Networking: HTTP',
        'Networking: DNS', 'Networking: IMAP', 'Networking: Cookies',
        'WebRTC: Networking', 'Networking: WebSockets', 'Networking: POP',
        'Networking: SMTP', 'Networking: Cache', 'Networking: NNTP',
        'Networking: Proxy', 'Networking: JAR', 'Networking: File',
        'Networking: FTP', 'DOM: Service Workers'],
    'Accessibility': [
        'Disability Access APIs', 'Disability Access', 'Accessibility Tools'],
    'Performance': [
        'Performance', 'DOM: Performance', 'Perfherder', 'mozperftest',
        'Performance Tools (Profiler/Timeline)', 'Performance Monitoring',
        'PerfCompare', 'Performance Engineering', 'Raptor'],
    'Storage and Data': [
        'Storage: IndexedDB', 'Form Autofill', 'Address Book'],
}
mapping = {comp: label for label, comps in groups.items() for comp in comps}

keep = {'full', 'down', 'up', 'top', 'bottom', 'front', 'back', 'side', 'system', 'empty'}
extra = {'doesn', 'don', 'isn', 'didn', 'wasn', 'aren', 'weren', 'couldn',
         'wouldn', 'shouldn', 'hasn', 'haven', 'hadn', 'won', 'll', 've', 're'}
stop = (set(ENGLISH_STOP_WORDS) - keep) | extra

def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-z]+', ' ', text)
    words = [w for w in text.split() if len(w) > 1 and w not in stop]
    return ' '.join(words)

cols = ['component', 'product', 'summary', 'type']
df = pd.read_csv(RAW, usecols=cols, low_memory=False)
print('Step 1 - loaded:', len(df))

df = df[df['type'] == 'defect']
print('Step 2 - defects only:', len(df))

df = df[df['product'] != 'Invalid Bugs']
print('Step 3 - without Invalid Bugs:', len(df))

df = df[~df['component'].isin(['General', 'Untriaged'])]
print('Step 4 - without General and Untriaged:', len(df))

df['label'] = df['component'].map(mapping)
df = df.dropna(subset=['label'])
print('Step 5 - 8 categories:', len(df))

df = df.dropna(subset=['summary'])
print('Step 6 - no empty summaries:', len(df))

df = df.drop_duplicates(subset=['summary', 'label'])
print('Step 7 - without duplicate titles:', len(df))

df[['label', 'summary']].to_csv(LABELED, index=False)
print('Step 8 - saved labeled file:', LABELED)

df['clean'] = df['summary'].astype(str).apply(clean)
print('Step 9 - text cleaned:', len(df))

df = df[df['clean'] != '']
print('Step 10 - without empty cleaned titles:', len(df))

before = df['label'].value_counts()

df = df.drop_duplicates(subset=['clean', 'label'])
print('Step 11 - without duplicates after cleaning:', len(df))

after = df['label'].value_counts()

table = pd.DataFrame({'before duplicates': before, 'after duplicates': after})
table['removed'] = table['before duplicates'] - table['after duplicates']
print(table)

df[['label', 'clean']].to_csv(OUT, index=False)
print('Step 12 - saved clean file:', OUT)
print(df[['summary', 'clean']].head(10).to_string())