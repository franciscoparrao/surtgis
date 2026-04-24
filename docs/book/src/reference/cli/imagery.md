# `surtgis imagery`

_Imagery / spectral index algorithms_

## Overview

```text
Imagery / spectral index algorithms

Usage: surtgis imagery [OPTIONS] <COMMAND>

Commands:
  ndvi              NDVI: Normalized Difference Vegetation Index
  ndwi              NDWI: Normalized Difference Water Index
  mndwi             MNDWI: Modified Normalized Difference Water Index
  nbr               NBR: Normalized Burn Ratio
  savi              SAVI: Soil-Adjusted Vegetation Index
  evi               EVI: Enhanced Vegetation Index
  bsi               BSI: Bare Soil Index
  band-math         Band math: arithmetic between two raster bands
  calc              Compute custom spectral index from arithmetic expression
  evi2              EVI2: Two-band Enhanced Vegetation Index
  gndvi             GNDVI: Green Normalized Difference Vegetation Index
  ngrdi             NGRDI: Normalized Green-Red Difference Index
  reci              RECI: Red Edge Chlorophyll Index
  ndre              NDRE: Normalized Difference Red Edge Index
  ndsi              NDSI: Normalized Difference Snow Index
  ndmi              NDMI: Normalized Difference Moisture Index
  ndbi              NDBI: Normalized Difference Built-up Index
  msavi             MSAVI: Modified Soil-Adjusted Vegetation Index
  reclassify        Reclassify raster values into discrete classes
  median-composite  Per-pixel median composite across multiple rasters
  dnbr              dNBR: differenced Normalized Burn Ratio (pre/post fire)
  cloud-mask        Cloud mask using Sentinel-2 SCL band
  help              Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery ndvi` {#ndvi}

```text
NDVI: Normalized Difference Vegetation Index

Usage: surtgis imagery ndvi [OPTIONS] --nir <NIR> --red <RED> <OUTPUT>

Arguments:
  <OUTPUT>  Output file

Options:
      --nir <NIR>                NIR band file
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --red <RED>                Red band file
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery ndwi` {#ndwi}

```text
NDWI: Normalized Difference Water Index

Usage: surtgis imagery ndwi [OPTIONS] --green <GREEN> --nir <NIR> <OUTPUT>

Arguments:
  <OUTPUT>  Output file

Options:
      --green <GREEN>            Green band file
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --nir <NIR>                NIR band file
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery mndwi` {#mndwi}

```text
MNDWI: Modified Normalized Difference Water Index

Usage: surtgis imagery mndwi [OPTIONS] --green <GREEN> --swir <SWIR> <OUTPUT>

Arguments:
  <OUTPUT>  Output file

Options:
      --green <GREEN>            Green band file
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --swir <SWIR>              SWIR band file
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery nbr` {#nbr}

```text
NBR: Normalized Burn Ratio

Usage: surtgis imagery nbr [OPTIONS] --nir <NIR> --swir <SWIR> <OUTPUT>

Arguments:
  <OUTPUT>  Output file

Options:
      --nir <NIR>                NIR band file
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --swir <SWIR>              SWIR band file
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery savi` {#savi}

```text
SAVI: Soil-Adjusted Vegetation Index

Usage: surtgis imagery savi [OPTIONS] --nir <NIR> --red <RED> <OUTPUT>

Arguments:
  <OUTPUT>  Output file

Options:
      --nir <NIR>                NIR band file
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --red <RED>                Red band file
  -l, --l-factor <L_FACTOR>      Soil brightness correction factor (0..1) [default: 0.5]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery evi` {#evi}

```text
EVI: Enhanced Vegetation Index

Usage: surtgis imagery evi [OPTIONS] --nir <NIR> --red <RED> --blue <BLUE> <OUTPUT>

Arguments:
  <OUTPUT>  Output file

Options:
      --nir <NIR>                NIR band file
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --red <RED>                Red band file
      --blue <BLUE>              Blue band file
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery bsi` {#bsi}

```text
BSI: Bare Soil Index

Usage: surtgis imagery bsi [OPTIONS] --swir <SWIR> --red <RED> --nir <NIR> --blue <BLUE> <OUTPUT>

Arguments:
  <OUTPUT>  Output file

Options:
      --swir <SWIR>              SWIR band file
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --red <RED>                Red band file
      --nir <NIR>                NIR band file
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --blue <BLUE>              Blue band file
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery band-math` {#band-math}

```text
Band math: arithmetic between two raster bands

Usage: surtgis imagery band-math [OPTIONS] -a <A> -b <B> --op <OP> <OUTPUT>

Arguments:
  <OUTPUT>  Output file

Options:
  -a <A>                         First input raster
  -v, --verbose                  Verbose output
  -b <B>                         Second input raster
      --compress                 Compress output GeoTIFFs (deflate)
      --op <OP>                  Operation: add, subtract, multiply, divide, power, min, max
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery calc` {#calc}

```text
Compute custom spectral index from arithmetic expression

Common formulas: NDVI:  "(NIR - Red) / (NIR + Red)" EXG:   "2 * Green - Red - Blue" VARI:  "(Green - Red) / (Green + Red - Blue)" Clay:  "SWIR1 / SWIR2"

Usage: surtgis imagery calc [OPTIONS] --expression <EXPRESSION> <OUTPUT>

Arguments:
  <OUTPUT>
          Output file

Options:
  -e, --expression <EXPRESSION>
          Arithmetic expression using band names

  -v, --verbose
          Verbose output

  -b, --band <NAME=FILE>
          Band assignments as NAME=path (repeatable)

      --compress
          Compress output GeoTIFFs (deflate)

      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)

      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming

  -h, --help
          Print help (see a summary with '-h')
```

## `imagery evi2` {#evi2}

```text
EVI2: Two-band Enhanced Vegetation Index

Usage: surtgis imagery evi2 [OPTIONS] --nir <NIR> --red <RED> <OUTPUT>

Arguments:
  <OUTPUT>  

Options:
      --nir <NIR>                
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --red <RED>                
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery gndvi` {#gndvi}

```text
GNDVI: Green Normalized Difference Vegetation Index

Usage: surtgis imagery gndvi [OPTIONS] --nir <NIR> --green <GREEN> <OUTPUT>

Arguments:
  <OUTPUT>  

Options:
      --nir <NIR>                
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --green <GREEN>            
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery ngrdi` {#ngrdi}

```text
NGRDI: Normalized Green-Red Difference Index

Usage: surtgis imagery ngrdi [OPTIONS] --green <GREEN> --red <RED> <OUTPUT>

Arguments:
  <OUTPUT>  

Options:
      --green <GREEN>            
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --red <RED>                
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery reci` {#reci}

```text
RECI: Red Edge Chlorophyll Index

Usage: surtgis imagery reci [OPTIONS] --nir <NIR> --red-edge <RED_EDGE> <OUTPUT>

Arguments:
  <OUTPUT>  

Options:
      --nir <NIR>                
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --red-edge <RED_EDGE>      
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery ndre` {#ndre}

```text
NDRE: Normalized Difference Red Edge Index

Usage: surtgis imagery ndre [OPTIONS] --nir <NIR> --red-edge <RED_EDGE> <OUTPUT>

Arguments:
  <OUTPUT>  

Options:
      --nir <NIR>                
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --red-edge <RED_EDGE>      
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery ndsi` {#ndsi}

```text
NDSI: Normalized Difference Snow Index

Usage: surtgis imagery ndsi [OPTIONS] --green <GREEN> --swir <SWIR> <OUTPUT>

Arguments:
  <OUTPUT>  

Options:
      --green <GREEN>            
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --swir <SWIR>              
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery ndmi` {#ndmi}

```text
NDMI: Normalized Difference Moisture Index

Usage: surtgis imagery ndmi [OPTIONS] --nir <NIR> --swir <SWIR> <OUTPUT>

Arguments:
  <OUTPUT>  

Options:
      --nir <NIR>                
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --swir <SWIR>              
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery ndbi` {#ndbi}

```text
NDBI: Normalized Difference Built-up Index

Usage: surtgis imagery ndbi [OPTIONS] --swir <SWIR> --nir <NIR> <OUTPUT>

Arguments:
  <OUTPUT>  

Options:
      --swir <SWIR>              
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --nir <NIR>                
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery msavi` {#msavi}

```text
MSAVI: Modified Soil-Adjusted Vegetation Index

Usage: surtgis imagery msavi [OPTIONS] --nir <NIR> --red <RED> <OUTPUT>

Arguments:
  <OUTPUT>  

Options:
      --nir <NIR>                
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --red <RED>                
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery reclassify` {#reclassify}

```text
Reclassify raster values into discrete classes

Usage: surtgis imagery reclassify [OPTIONS] <INPUT> <OUTPUT>

Arguments:
  <INPUT>   
  <OUTPUT>  

Options:
      --class <MIN,MAX,VALUE>    Class definition as "min,max,value" (repeatable)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --default <DEFAULT>        Default value for unclassified cells [default: NaN]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery median-composite` {#median-composite}

```text
Per-pixel median composite across multiple rasters

Usage: surtgis imagery median-composite [OPTIONS] <OUTPUT>

Arguments:
  <OUTPUT>  Output file

Options:
  -i, --input <INPUT>            Input raster files (at least 2, repeatable)
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

## `imagery dnbr` {#dnbr}

```text
dNBR: differenced Normalized Burn Ratio (pre/post fire)

Usage: surtgis imagery dnbr [OPTIONS] --pre-nir <PRE_NIR> --pre-swir <PRE_SWIR> --post-nir <POST_NIR> --post-swir <POST_SWIR> <OUTPUT>

Arguments:
  <OUTPUT>  

Options:
      --pre-nir <PRE_NIR>
          
  -v, --verbose
          Verbose output
      --compress
          Compress output GeoTIFFs (deflate)
      --pre-swir <PRE_SWIR>
          
      --post-nir <POST_NIR>
          
      --streaming
          Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>
          Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
      --post-swir <POST_SWIR>
          
      --severity-output <SEVERITY_OUTPUT>
          Also output burn severity classification
  -h, --help
          Print help
```

## `imagery cloud-mask` {#cloud-mask}

```text
Cloud mask using Sentinel-2 SCL band

Usage: surtgis imagery cloud-mask [OPTIONS] --scl <SCL> <INPUT> <OUTPUT>

Arguments:
  <INPUT>   Input raster to mask
  <OUTPUT>  Output file

Options:
      --scl <SCL>                SCL classification raster
  -v, --verbose                  Verbose output
      --compress                 Compress output GeoTIFFs (deflate)
      --keep <KEEP>              SCL classes to keep (comma-separated, default: 4,5,6,11) [default: 4,5,6,11]
      --streaming                Force streaming mode for large rasters (auto-detected if >500MB)
      --max-memory <MAX_MEMORY>  Maximum memory to use (e.g., 4G, 1024MB, 500MiB). If raster would exceed this when decompressed, force streaming
  -h, --help                     Print help
```

