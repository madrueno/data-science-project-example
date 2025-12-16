# Ejemplo Proyectos PLN

Este repositorio muestra un ejemplo educativo para el desarrollo de proyectos de procesamiento del lenguaje natural.



## Estructura del repositorio

La siguiente estructura organiza el proyecto de forma clara y reproducible, siguiendo buenas prácticas en proyectos de procesamiento de lenguaje natural.

```bash
.
├── data/           # datos organizados por etapas de transformacion
│   ├── raw/        # datos originales en crudo sin modificar
│   ├── interim/    # datos con transformaciones intermedias
│   └── processed/  # datos listos para experimentacion
├── models/         # modelos entrenados y artefactos
├── notebooks/      # notebooks de exploracion
├── reports/        # documentacion elaborada
├── runs/           # ejecuciones del proyecto
├── src/            # codigo principal del proyecto
│   ├── cli/        # comandos de consola para experimentacion
│   ├── config/     # configuracion general
│   ├── datasets/   # preparacion de datos
│   ├── evaluation/ # evaluacion de modelos
│   ├── modeling/   # definicion de modelos
│   └── pipelines/  # pipelines de experimentacion
├── .env.sample     # placeholder de variables de entorno
├── .gitignore      # archivos y carpetas sin trackear
├── README.md       # guia del repositorio
├── LICENSE         # licencia del repositorio
├── pyproject.toml  # dependencias y configuracion del proyecto
└── uv.lock         # fijado de versiones para reproducibilidad
```



## Preparación del proyecto

Pasos básicos para configurar y ejecutar el proyecto localmente.


### Gestión dependencias

El proyecto utiliza `uv` para la gestión de dependencias y del entorno virtual. Algunos comandos básicos son:

```bash
uv sync              # sincronizar dependencias y crear entorno virtual
uv add <paquete>     # anadir una dependencia
uv remove <paquete>  # eliminar dependencias
uv run jupyter lab   # ejecutar notebooks
```

Más información en:
- [Documentación oficial de uv](https://docs.astral.sh/uv/)
- [Instalación de uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Guía de comandos de uv](https://docs.astral.sh/uv/reference/cli/)
- [Gestión de proyectos con uv](https://docs.astral.sh/uv/guides/projects/) 


### Variables de entorno

Conviene guardar las variables que no queremos versionar en Git en un fichero `.env`. Es buena práctica subir un archivo de ejemplo con las variables a rellenar (como `.env.sample`) y rellenarlo localmente en un `.env`.
```bash
cp .env.sample .env # rellenar variables localmente
```



## Ejecución
TODO