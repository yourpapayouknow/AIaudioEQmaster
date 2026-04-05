    param(
      [Parameter(Mandatory=$true)][string]$Input,
      [Parameter(Mandatory=$true)][string]$Output,
      [string]$Genre = "Pop",
      [string]$Style = "Fusion",
      [ValidateSet("soft","dynamic","normal","loud")][string]$Loudness = "loud"
    )

    python main.py $Input $Output --genre $Genre --eq-profile $Style --loudness $Loudness
