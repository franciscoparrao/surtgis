Feature Request: Soporte para datasets climáticos en STAC (Zarr/NetCDF)                                                                                                             
                                                                                                                                                                                        
    Contexto                                                                                                                                                                            
                                                                                                                                                                                        
    Al intentar usar surtgis stac composite para descargar datos climáticos desde Planetary Computer, encontramos que las tres colecciones relevantes usan formatos incompatibles con el
     pipeline actual de SurtGis (que espera COG/GeoTIFF):                                                                                                                               
                                                                                                                                                                                        
    ┌─────────────────────┬───────────────────────────────────────────────┬─────────────┬──────────┐                                                                                    
    │      Colección      │                      Uso                      │   Formato   │ SurtGis? │                                                                                    
    ├─────────────────────┼───────────────────────────────────────────────┼─────────────┼──────────┤                                                                                    
    │ era5-pds            │ Reanalysis histórico (precip, temp, 1979-hoy) │ Zarr        │ No       │                                                                                    
    ├─────────────────────┼───────────────────────────────────────────────┼─────────────┼──────────┤                                                                                    
    │ nasa-nex-gddp-cmip6 │ Proyecciones SSP (26 GCMs, 2015-2100)         │ NetCDF      │ No       │                                                                                    
    ├─────────────────────┼───────────────────────────────────────────────┼─────────────┼──────────┤                                                                                    
    │ terraclimate        │ Clima mensual (~4km, 1958-2021)               │ Zarr/NetCDF │ No       │                                                                                    
    └─────────────────────┴───────────────────────────────────────────────┴─────────────┴──────────┘                                                                                    
                                                                                                                                                                                        
    Por qué importa                                                                                                                                                                     
                                                                                                                                                                                        
    Estos datasets son esenciales para análisis de susceptibilidad bajo cambio climático. El pipeline típico sería:                                                                     
                                                                                                                                                                                        
    # Ideal: descargar proyección CMIP6 para una cuenca                                                                                                                                 
    surtgis stac composite \                                                                                                                                                            
      --catalog pc \                                                                                                                                                                    
      --collection nasa-nex-gddp-cmip6 \                                                                                                                                                
      --asset pr \                                                                                                                                                                      
      --bbox "-71.0,-29.0,-70.5,-28.5" \                                                                                                                                                
      --datetime "2050-01-01/2050-12-31" \                                                                                                                                              
      output_precip_2050.tif                                                                                                                                                            
                                                                                                                                                                                        
    # O ERA5 histórico                                                                                                                                                                  
    surtgis stac composite \                                                                                                                                                            
      --catalog pc \                                                                                                                                                                    
      --collection era5-pds \                                                                                                                                                           
      --asset precipitation_amount_1hour_Accumulation \                                                                                                                                 
                                                                                            
  ↑/↓ to scroll · Space, Enter, or Escape to dismiss

 # O ERA5 histórico                                                                                                                                                                  
    surtgis stac composite \                                                                                                                                                            
      --catalog pc \                                                                                                                                                                    
      --collection era5-pds \                                                                                                                                                           
      --asset precipitation_amount_1hour_Accumulation \                                                                                                                                 
      --bbox "-71.0,-29.0,-70.5,-28.5" \                                                                                                                                                
      --datetime "2020-01-01/2020-12-31" \                                                                                                                                              
      output_era5_precip_2020.tif                                                                                                                                                       
                                                                                                                                                                                        
    Actualmente esto falla porque SurtGis solo sabe leer COGs, no Zarr ni NetCDF.                                                                                                       
                                                                                                                                                                                        
    Datasets afectados en Planetary Computer                                                                                                                                            
                                                                                                                                                                                        
    - ERA5-PDS (era5-pds): 4 variables, formato application/vnd+zarr                                                                                                                    
    - NEX-GDDP-CMIP6 (nasa-nex-gddp-cmip6): 7 variables (pr, tas, hurs, huss, rlds, rsds, sfcWind), formato application/netcdf, 26 modelos GCM, escenarios SSP 2-4.5 y SSP 5-8.5,       
    resolución ~25km                                                                                                                                                                    
    - TerraClimate (terraclimate): mensual, ~4km                                                                                                                                        
                                                                                                                                                                                        
    Propuesta de implementación                                                                                                                                                         
                                                                                                                                                                                        
    Opción A: Soporte NetCDF/Zarr en lectura (más completo)                                                                                                                             
                                                                                                                                                                                        
    Agregar readers para NetCDF (via netcdf3 crate o bindings a HDF5) y Zarr. Cuando stac composite detecta que el asset es NetCDF/Zarr en vez de COG:                                  
    1. Descargar el archivo completo (son ~50-200MB por variable por año)                                                                                                               
    2. Extraer la variable solicitada                                                                                                                                                   
    3. Recortar al bbox                                                                                                                                                                 
    4. Convertir a GeoTIFF internamente                                                                                                                                                 
    5. Continuar con el pipeline normal de compositing                

Opción B: Conversión on-the-fly (más simple)                                                                                                                                        
                                                                                                                                                                                 
    Nuevo subcomando:                                                                               
    surtgis stac download-climate \                                                                                                                                                     
      --catalog pc \                                                                                                                                                             
      --collection nasa-nex-gddp-cmip6 \                                                            
      --variable pr \                                                                                                                                                                   
      --model "MIROC6,MPI-ESM1-2-HR,GFDL-ESM4"  # multi-model ensemble                                                                                                           
      --scenario ssp245 \                                                                           
      --bbox "-71.0,-29.0,-70.5,-28.5" \                                                                                                              
      --datetime "2030-01-01/2050-12-31" \                                                                                                                                       
      --aggregate yearly-mean \                                                                      
      output_dir/                                                                                                                                     
                                                                                                                                                                                 
    Que internamente: descarga NetCDF → extrae bbox → agrega temporalmente → exporta GeoTIFF por año.
                                                                                                                                                                                 
    Opción C: Wrapper mínimo (workaround)                                                                                                                                        
                                                                                                     
    Nuevo flag --format netcdf en stac composite que use xarray/Python como backend de lectura en vez del lector COG nativo. Menos performante pero reutiliza toda la lógica STAC
    existente.                                                                                                                                                                   
                                                                                 
    Datos disponibles en NEX-GDDP-CMIP6                                                                                                                                          
                                                                                                                                                                                 
    Para referencia, lo que encontramos explorando el catálogo:                     
                                                                                                                                                      
    - 26 modelos GCM: MIROC6, MPI-ESM1-2-HR, GFDL-ESM4, EC-Earth3, UKESM1-0-LL, etc.                                                                                             
    - Escenarios: historical (pre-2015), ssp245, ssp585                             
    - Variables: pr (precipitación), tas (temperatura), hurs (humedad), sfcWind (viento)                                                              
    - Resolución: ~0.25° (~25km)                                                                                                                                                 
    - Cobertura: global, 1950-2100                                                  
    - Estructura item ID: {MODEL}.{SCENARIO}.{YEAR} (ej: MIROC6.ssp245.2050)                                                                          
                                                                                                                                                                                   
    Impacto                                                                                                                                                                        
                                                                                                                                                      
    Sin este soporte, los usuarios deben salir de SurtGis y usar Python (xarray + planetary_computer + netCDF4) para datos climáticos, rompiendo el pipeline unificado. Dado que el
    análisis de cambio climático es cada vez más central en geociencias, tener soporte nativo sería un diferenciador importante para la herramienta.       