"""Verify selected public materials without any external access."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parent
index=json.loads((root/'material_manifest.json').read_text(encoding='utf-8'))
for name,record in index['files'].items():
    path=root/name
    assert path.resolve().is_relative_to(root.resolve()),name
    data=path.read_bytes()
    assert len(data)==record['bytes'],name
    assert hashlib.sha256(data).hexdigest()==record['sha256'],name
print(json.dumps({'status':'passed','selected_material_files':len(index['files']),
    'numeric_CSV_observations_unchanged':index['scientific_numeric_CSV_observations_unchanged'],
    'network_or_hardware_access':False}))
