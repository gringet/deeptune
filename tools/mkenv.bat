@setlocal
@set "HERE=%~dp0"
@pushd "%HERE%"

if exist ../env rmdir /s/q ../env
python -m venv --prompt "deeptune" ../env
call ..\env\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r ../requirements.txt
deactivate

@popd
@endlocal
