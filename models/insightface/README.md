# InsightFace Model Directory

This project expects local InsightFace models here:

```text
models/insightface/models/buffalo_l/
models/insightface/models/antelopev2/
```

The ONNX files are intentionally ignored by git because they are large binary model files. On this machine they were copied from:

```text
D:\ComfyUI\models\insightface\models\buffalo_l
D:\ComfyUI\models\insightface\models\antelopev2
```

To restore the local model layout on Windows:

```bat
mkdir E:\Face-similarity\models\insightface\models\buffalo_l
mkdir E:\Face-similarity\models\insightface\models\antelopev2
xcopy /E /I /Y D:\ComfyUI\models\insightface\models\buffalo_l E:\Face-similarity\models\insightface\models\buffalo_l
xcopy /E /I /Y D:\ComfyUI\models\insightface\models\antelopev2 E:\Face-similarity\models\insightface\models\antelopev2
```

