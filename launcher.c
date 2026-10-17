/* QuixelStylizer.exe

   A windowed host so the taskbar can pin this program instead of python.exe.
   Python itself stays the install in LocalAppData. The .py files next to this
   exe are what actually run, so editing them does not require a rebuild.

   python312.dll is delay-loaded. SetDllDirectory has to run before the first
   Python call, or the loader looks next to this exe and fails.
*/
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <stdlib.h>

static void fail(const wchar_t *msg) {
    MessageBoxW(NULL, msg, L"Quixel Stylizer", MB_ICONERROR | MB_OK);
}

static int find_python(wchar_t *home, size_t cap) {
    const wchar_t *tails[] = {
        L"\\Programs\\Python\\Python312",
        L"\\Python312",
    };
    const wchar_t *roots[] = { L"LOCALAPPDATA", L"ProgramFiles" };
    for (int i = 0; i < 2; i++) {
        wchar_t root[MAX_PATH];
        DWORD n = GetEnvironmentVariableW(roots[i], root, MAX_PATH);
        if (n == 0 || n >= MAX_PATH)
            continue;
        _snwprintf_s(home, cap, _TRUNCATE, L"%s%s", root, tails[i]);
        wchar_t dll[MAX_PATH];
        _snwprintf_s(dll, MAX_PATH, _TRUNCATE, L"%s\\python312.dll", home);
        if (GetFileAttributesW(dll) != INVALID_FILE_ATTRIBUTES)
            return 1;
    }
    return 0;
}

int WINAPI WinMain(HINSTANCE inst, HINSTANCE prev, LPSTR cmd, int show) {
    (void)inst;
    (void)prev;
    (void)cmd;
    (void)show;

    /* Before any window exists, including the ones Tk creates later. */
    HMODULE shell = LoadLibraryW(L"shell32.dll");
    if (shell) {
        typedef HRESULT(WINAPI *SetIdFn)(PCWSTR);
        SetIdFn set_id = (SetIdFn)GetProcAddress(shell, "SetCurrentProcessExplicitAppUserModelID");
        if (set_id)
            set_id(L"InvertedGames.QuixelStylizer");
    }
    HMODULE shcore = LoadLibraryW(L"shcore.dll");
    if (shcore) {
        typedef HRESULT(WINAPI *SetDpiFn)(int);
        SetDpiFn set_dpi = (SetDpiFn)GetProcAddress(shcore, "SetProcessDpiAwareness");
        if (set_dpi)
            set_dpi(1);
    }

    wchar_t home[MAX_PATH];
    if (!find_python(home, MAX_PATH)) {
        fail(L"Python 3.12 was not found.\n\n"
             L"Install it with:\n"
             L"  winget install -e --id Python.Python.3.12\n"
             L"then run bootstrap.ps1 next to this program.");
        return 1;
    }
    SetDllDirectoryW(home);
    SetEnvironmentVariableW(L"PYTHONDONTWRITEBYTECODE", L"1");

    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(NULL, exe, MAX_PATH);
    wchar_t dir[MAX_PATH];
    wcscpy_s(dir, MAX_PATH, exe);
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (slash)
        *slash = 0;

    wchar_t script[MAX_PATH];
    _snwprintf_s(script, MAX_PATH, _TRUNCATE, L"%s\\run.py", dir);
    if (GetFileAttributesW(script) == INVALID_FILE_ATTRIBUTES) {
        fail(L"run.py is not next to QuixelStylizer.exe.\nRun bootstrap.ps1 if the install is incomplete.");
        return 1;
    }

    int argc = 0;
    wchar_t **full = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!full || argc < 1) {
        fail(L"Could not read the command line.");
        return 1;
    }
    wchar_t **py_argv = (wchar_t **)calloc((size_t)argc, sizeof(wchar_t *));
    if (!py_argv) {
        LocalFree(full);
        return 1;
    }
    py_argv[0] = script;
    for (int i = 1; i < argc; i++)
        py_argv[i] = full[i];

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    config.parse_argv = 0;

    PyStatus status = PyConfig_SetString(&config, &config.home, home);
    if (PyStatus_Exception(status))
        goto pyfail;
    status = PyConfig_SetString(&config, &config.program_name, exe);
    if (PyStatus_Exception(status))
        goto pyfail;
    status = PyConfig_SetArgv(&config, argc, py_argv);
    if (PyStatus_Exception(status))
        goto pyfail;
    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status))
        goto pyfail;
    PyConfig_Clear(&config);
    LocalFree(full);
    free(py_argv);
    full = NULL;
    py_argv = NULL;

    /* sys.argv[0] is the script. run_path is the same entry the .bat uses. */
    int rc = PyRun_SimpleString(
        "import runpy, sys\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n");
    Py_FinalizeEx();
    if (rc != 0) {
        fail(L"Quixel Stylizer failed to start.\nThe details are in stylizer.log next to this program.");
        return 1;
    }
    return 0;

pyfail: {
    const char *err = status.err_msg ? status.err_msg : "Python failed to start";
    wchar_t wide[512];
    MultiByteToWideChar(CP_UTF8, 0, err, -1, wide, 512);
    fail(wide);
    PyConfig_Clear(&config);
    if (full)
        LocalFree(full);
    free(py_argv);
    return 1;
}
}
