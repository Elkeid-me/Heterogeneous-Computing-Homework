open System
open System.Runtime.InteropServices
open SixLabors.ImageSharp
open SixLabors.ImageSharp.PixelFormats

let baseDir = AppDomain.CurrentDomain.BaseDirectory

module Native =
    [<Literal>]
    let LibraryName = "rotate_image_opencl"

    [<DllImport(LibraryName, EntryPoint = "rotate_image_opencl", CallingConvention = CallingConvention.Cdecl)>]
    extern void rotate_image_opencl(byte[], byte[], int, int, single)

    [<DllImport(LibraryName, EntryPoint = "rotate_image_cpu", CallingConvention = CallingConvention.Cdecl)>]
    extern void rotate_image_cpu(byte[], byte[], int, int, single)

type CliOptions =
    { InputPath: string
      OutputPath: string
      Degrees: single
      Engine: string }

let usage () =
    eprintfn "Usage: RotateBmpFs -i <input image> -o <output image> -d <degrees> [-e cpu|opencl]"
    exit 1

let parseArgs (args: string[]) =
    let rec parseArgsImpl args argsSoFar =
        match args with
        | [] -> argsSoFar
        | "-i" :: value :: rest -> parseArgsImpl rest { argsSoFar with InputPath = value }
        | "-o" :: value :: rest -> parseArgsImpl rest { argsSoFar with OutputPath = value }
        | "-d" :: value :: rest ->

            parseArgsImpl
                rest
                { argsSoFar with
                    Degrees = single value }
        | "-e" :: value :: rest ->
            parseArgsImpl
                rest
                { argsSoFar with
                    Engine = value.ToLower() }
        | _ -> usage ()

    parseArgsImpl
        (List.ofArray args)
        { InputPath = ""
          OutputPath = ""
          Degrees = 0.0f
          Engine = "opencl" }

let loadPixels (path: string) =
    use image = Image.Load<Rgba32> path
    let pixels = Array.zeroCreate<byte> (image.Width * image.Height * 4)
    image.CopyPixelDataTo(pixels.AsSpan())
    image.Width, image.Height, pixels

let savePixels (path: string) width height (pixels: byte[]) =
    use image = Image.LoadPixelData<Rgba32>(pixels, width, height)
    image.Save path

let rotateCpu (src: byte[]) (width: int) (height: int) (degrees: single) =
    let dst = Array.zeroCreate<byte> src.Length
    Native.rotate_image_cpu (src, dst, width, height, degrees)
    dst

let rotateOpenCl (src: byte[]) (width: int) (height: int) (degrees: single) =
    let dst = Array.zeroCreate<byte> src.Length
    Native.rotate_image_opencl (src, dst, width, height, degrees)
    dst

[<EntryPoint>]
let main argv =
    let args = parseArgs argv

    let width, height, srcPixels = loadPixels args.InputPath

    let outputPixels =
        match args.Engine with
        | "cpu" -> rotateCpu srcPixels width height args.Degrees
        | "opencl" -> rotateOpenCl srcPixels width height args.Degrees
        | other -> failwith $"Unknown engine: {other}"

    savePixels args.OutputPath width height outputPixels

    0
