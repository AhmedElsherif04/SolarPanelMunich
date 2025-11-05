import xml.etree.ElementTree as ET
from collections import Counter

path = "/home/elsherif/Desktop/Thesis/Lod2/Neuberg/662_5400.gml"

# Parse GML
tree = ET.parse(path)
root = tree.getroot()

# Get namespaces
namespaces = dict([
    node for _, node in ET.iterparse(path, events=['start-ns'])
])

# Find the "bldg" namespace
bldg_prefix = None
for k, v in namespaces.items():
    if "building" in v.lower():
        bldg_prefix = k
        break

if not bldg_prefix:
    raise ValueError("❌ Could not find building namespace.")

bldg_ns = namespaces[bldg_prefix]
buildings = root.findall(f".//{{{bldg_ns}}}Building")

print(f"✅ Found {len(buildings)} buildings\n")

# Collect all attribute names and child tags across all buildings
attr_counter = Counter()
child_tag_counter = Counter()

for b in buildings:
    for attr in b.attrib.keys():
        attr_counter[attr] += 1
    for child in b:
        tag = child.tag.split('}')[-1]  # remove namespace
        child_tag_counter[tag] += 1

# ✅ Print results
print("📘 Unique Building Attributes:")
for attr, count in attr_counter.items():
    print(f"  - {attr}  (appears in {count} buildings)")

print("\n🏷️ Unique Child Tags inside <Building>:")
for tag, count in child_tag_counter.items():
    print(f"  - {tag}  (appears in {count} buildings)")
