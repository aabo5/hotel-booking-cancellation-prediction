# Exports the final report (.docx) and presentation (.pptx) in report\ to PDF using installed Office.
# Usage: powershell -File tools\export_office.ps1 [-Png <folder>]   (-Png also saves page/slide images for checking)
param([string]$Png = "")
$report = Join-Path (Split-Path $PSScriptRoot -Parent) "report"

$pptx = Get-ChildItem $report -Filter "Final_*.pptx"
if ($pptx) {
    $ppt = New-Object -ComObject PowerPoint.Application
    foreach ($f in $pptx) {
        $p = $ppt.Presentations.Open($f.FullName, $true, $false, $false)
        $pdf = [IO.Path]::ChangeExtension($f.FullName, ".pdf")
        $p.SaveAs($pdf, 32)
        if ($Png) {
            $dir = Join-Path $Png $f.BaseName
            New-Item -ItemType Directory -Force $dir | Out-Null
            foreach ($s in $p.Slides) { $s.Export((Join-Path $dir ("slide{0:D2}.png" -f $s.SlideIndex)), "PNG", 1600, 900) }
        }
        $p.Close()
        Write-Output "PDF: $pdf"
    }
    $ppt.Quit()
}

$docs = Get-ChildItem $report -Filter "Final_*.docx" | Where-Object { $_.Name -notlike '~$*' }
if ($docs) {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    foreach ($d in $docs) {
        $doc = $word.Documents.Open($d.FullName, $false, $false)
        if ($doc.TablesOfContents.Count -gt 0) { $doc.TablesOfContents.Item(1).Update() }
        $doc.Fields.Update() | Out-Null
        $doc.Save()
        $pdf = [IO.Path]::ChangeExtension($d.FullName, ".pdf")
        $doc.SaveAs([ref]$pdf, [ref]17)
        Write-Output "PDF: $pdf  pages: $($doc.ComputeStatistics(2))"
        $doc.Close([ref]0)
    }
    $word.Quit()
}
