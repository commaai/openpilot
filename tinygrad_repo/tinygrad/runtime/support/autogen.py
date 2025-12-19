import ctypes.util, importlib.metadata, itertools, re, functools, os
from tinygrad.helpers import flatten, unwrap
from clang.cindex import Config, Index, CursorKind as CK, TranslationUnit as TU, LinkageKind as LK, TokenKind as ToK, TypeKind as TK
from clang.cindex import PrintingPolicy as PP, PrintingPolicyProperty as PPP, SourceRange

assert importlib.metadata.version('clang')[:2] == "20"
if not Config.loaded: Config.set_library_file(os.getenv("LIBCLANG_PATH", ctypes.util.find_library("clang-20")))

def fst(c): return next(c.get_children())
def last(c): return list(c.get_children())[-1]
def readext(f, fst, snd=None):
  with open(f, "r") as f:
    f.seek(start:=(fst.start.offset if isinstance(fst, SourceRange) else fst))
    return f.read((fst.end.offset if isinstance(fst, SourceRange) else snd)-start)
def attrs(c): return list(filter(lambda k: (v:=k.value) >= 400 and v < 500, map(lambda c: c.kind, c.get_children())))

base_rules = [(r'\s*\\\n\s*', ' '), (r'\s*\n\s*', ' '), (r'//.*', ''), (r'/\*.*?\*/', ''), (r'\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b', r'\1'),
              (r'\b0+(?=\d)', ''), (r'\s*&&\s*', r' and '), (r'\s*\|\|\s*', r' or '), (r'\s*!\s*', ' not '),
              (r'(struct|union|enum)\s*([a-zA-Z_][a-zA-Z0-9_]*\b)', r'\1_\2'),
              (r'\((unsigned )?(char|uint64_t)\)', ''), (r'^.*\d+:\d+.*$', ''), (r'^.*\w##\w.*$', '')]

ints = (TK.INT, TK.UINT, TK.LONG, TK.ULONG, TK.LONGLONG, TK.ULONGLONG)

def gen(dll, files, args=[], prolog=[], rules=[], epilog=[], recsym=False, use_errno=False, anon_names={}, types={}, parse_macros=True):
  macros, lines, anoncnt, types = [], [], itertools.count().__next__, {k:(v,True) for k,v in types.items()}
  def tname(t, suggested_name=None, typedef=None) -> str:
    suggested_name = anon_names.get(f"{(decl:=t.get_declaration()).location.file}:{decl.location.line}", suggested_name)
    nonlocal lines, types, anoncnt
    tmap = {TK.VOID:"None", TK.CHAR_U:"ctypes.c_ubyte", TK.UCHAR:"ctypes.c_ubyte", TK.CHAR_S:"ctypes.c_char", TK.SCHAR:"ctypes.c_char",
            **{getattr(TK, k):f"ctypes.c_{k.lower()}" for k in ["BOOL", "WCHAR", "FLOAT", "DOUBLE", "LONGDOUBLE"]},
            **{getattr(TK, k):f"ctypes.c_{'u' if 'U' in k else ''}int{sz}" for sz,k in
               [(16, "USHORT"), (16, "SHORT"), (32, "UINT"), (32, "INT"), (64, "ULONG"), (64, "LONG"), (64, "ULONGLONG"), (64, "LONGLONG")]}}

    if t.kind in tmap: return tmap[t.kind]
    if t.spelling in types and types[t.spelling][1]: return types[t.spelling][0]
    if ((f:=t).kind in (fks:=(TK.FUNCTIONPROTO, TK.FUNCTIONNOPROTO))) or (t.kind == TK.POINTER and (f:=t.get_pointee()).kind in fks):
      return f"ctypes.CFUNCTYPE({tname(f.get_result())}{(', '+', '.join(map(tname, f.argument_types()))) if f.kind==TK.FUNCTIONPROTO else ''})"
    match t.kind:
      case TK.POINTER: return "ctypes.c_void_p" if (ptr:=t.get_pointee()).kind == TK.VOID else f"ctypes.POINTER({tname(ptr)})"
      case TK.ELABORATED: return tname(t.get_named_type(), suggested_name)
      case TK.TYPEDEF if t.spelling == t.get_canonical().spelling: return tname(t.get_canonical())
      case TK.TYPEDEF:
        defined, nm = (canon:=t.get_canonical()).spelling in types, tname(canon, typedef=t.spelling.replace('::', '_'))
        types[t.spelling] = nm if t.spelling.startswith("__") else t.spelling.replace('::', '_'), True
        # RECORDs need to handle typedefs specially to allow for self-reference
        if canon.kind != TK.RECORD or defined: lines.append(f"{t.spelling.replace('::', '_')} = {nm}")
        return types[t.spelling][0]
      case TK.RECORD:
        # TODO: packed unions
        # TODO: pragma pack support
        # check for forward declaration
        if t.spelling in types: types[t.spelling] = (nm:=types[t.spelling][0]), len(list(t.get_fields())) != 0
        else:
          if decl.is_anonymous():
            types[t.spelling] = (nm:=(suggested_name or (f"_anon{'struct' if decl.kind == CK.STRUCT_DECL else 'union'}{anoncnt()}")), True)
          else: types[t.spelling] = (nm:=t.spelling.replace(' ', '_').replace('::', '_')), len(list(t.get_fields())) != 0
          lines.append(f"class {nm}({'Struct' if decl.kind==CK.STRUCT_DECL else 'ctypes.Union'}): pass")
          if typedef: lines.append(f"{typedef} = {nm}")
        acnt = itertools.count().__next__
        ll=["  ("+((fn:=f"'_{acnt()}'")+f", {tname(f.type, nm+fn[1:-1])}" if f.is_anonymous_record_decl() else f"'{f.spelling}', "+
            tname(f.type, f'{nm}_{f.spelling}'))+(f',{f.get_bitfield_width()}' if f.is_bitfield() else '')+")," for f in t.get_fields()]
        lines.extend(([f"{nm}._anonymous_ = ["+", ".join(f"'_{i}'" for i in range(n))+"]"] if (n:=acnt()) else [])+
                     ([f"{nm}._packed_ = True"] * (CK.PACKED_ATTR in attrs(decl)))+([f"{nm}._fields_ = [",*ll,"]"] if ll else []))
        return nm
      case TK.ENUM:
        # TODO: C++ and GNU C have forward declared enums
        if decl.is_anonymous(): types[t.spelling] = suggested_name or f"_anonenum{anoncnt()}", True
        else: types[t.spelling] = t.spelling.replace(' ', '_').replace('::', '_'), True
        lines.append(f"{types[t.spelling][0]} = CEnum({tname(decl.enum_type)})\n" +
                     "\n".join(f"{e.spelling} = {types[t.spelling][0]}.define('{e.spelling}', {e.enum_value})" for e in decl.get_children()
                     if e.kind == CK.ENUM_CONSTANT_DECL) + "\n")
        return types[t.spelling][0]
      case TK.CONSTANTARRAY:
        return f"({tname(t.get_array_element_type(), suggested_name.rstrip('s') if suggested_name else None)} * {t.get_array_size()})"
      case TK.INCOMPLETEARRAY: return f"({tname(t.get_array_element_type(), suggested_name.rstrip('s') if suggested_name else None)} * 0)"
      case _: raise NotImplementedError(f"unsupported type {t.kind}")

  for f in files:
    tu = Index.create().parse(f, args, options=TU.PARSE_DETAILED_PROCESSING_RECORD)
    (pp:=PP.create(tu.cursor)).set_property(PPP.TerseOutput, 1)
    for c in tu.cursor.walk_preorder():
      if str(c.location.file) != str(f) and (not recsym or c.kind not in (CK.FUNCTION_DECL,)): continue
      rollback = lines, types
      try:
        match c.kind:
          case CK.FUNCTION_DECL if c.linkage == LK.EXTERNAL and dll:
            # TODO: we could support name-mangling
            lines.append(f"# {c.pretty_printed(pp)}\ntry: ({c.spelling}:=dll.{c.spelling}).restype, {c.spelling}.argtypes = "
              f"{tname(c.result_type)}, [{', '.join(tname(arg.type) for arg in c.get_arguments())}]\nexcept AttributeError: pass\n")
          case CK.STRUCT_DECL | CK.UNION_DECL | CK.TYPEDEF_DECL | CK.ENUM_DECL: tname(c.type)
          case CK.MACRO_DEFINITION if parse_macros and len(toks:=list(c.get_tokens())) > 1:
            if toks[1].spelling == '(' and toks[0].extent.end.column == toks[1].extent.start.column:
              it = iter(toks[1:])
              _args = [t.spelling for t in itertools.takewhile(lambda t:t.spelling!=')', it) if t.kind == ToK.IDENTIFIER]
              if len(body:=list(it)) == 0: continue
              macros += [f"{c.spelling} = lambda {','.join(_args)}: {readext(f, body[0].location.offset, toks[-1].extent.end.offset)}"]
            else: macros += [f"{c.spelling} = {readext(f, toks[1].location.offset, toks[-1].extent.end.offset)}"]
          case CK.VAR_DECL if c.linkage == LK.INTERNAL:
            if (c.type.kind == TK.CONSTANTARRAY and c.type.get_array_element_type().get_canonical().kind in ints and
                (init:=last(c)).kind == CK.INIT_LIST_EXPR and all(re.match(r"\[.*\].*=", readext(f, c.extent)) for c in init.get_children())):
              cs = init.get_children()
              macros += [f"{c.spelling} = {{{','.join(f'{readext(f,next(it:=c.get_children()).extent)}:{readext(f,next(it).extent)}' for c in cs)}}}"]
            elif c.type.get_canonical().kind in ints: macros += [f"{c.spelling} = {readext(f, last(c).extent)}"]
            else: macros += [f"{c.spelling} = {tname(c.type)}({readext(f, last(c).extent)})"]
          case CK.VAR_DECL if c.linkage == LK.EXTERNAL and dll:
            lines.append(f"try: {c.spelling} = {tname(c.type)}.in_dll(dll, '{c.spelling}')\nexcept (ValueError,AttributeError): pass")
      except NotImplementedError as e:
        print(f"skipping {c.spelling}: {e}")
        lines, types = rollback
  main = (f"# mypy: ignore-errors\nimport ctypes{', os' if any('os' in s for s in dll) else ''}\n"
    "from tinygrad.helpers import unwrap\nfrom tinygrad.runtime.support.c import Struct, CEnum, _IO, _IOW, _IOR, _IOWR\n" + '\n'.join([*prolog,
      *(["from ctypes.util import find_library"]*any('find_library' in s for s in dll)),
      *(["def dll():",*flatten([[f"  try: return ctypes.CDLL(unwrap({d}){', use_errno=True' if use_errno else ''})",'  except: pass'] for d in dll]),
         "  return None", "dll = dll()\n"]*bool(dll)), *lines]) + '\n')
  macros = [r for m in macros if (r:=functools.reduce(lambda s,r:re.sub(r[0], r[1], s), rules + base_rules, m))]
  while True:
    try:
      exec(main + '\n'.join(macros), {})
      break
    except (SyntaxError, NameError, TypeError) as e:
      macrono = unwrap(e.lineno if isinstance(e, SyntaxError) else unwrap(unwrap(e.__traceback__).tb_next).tb_lineno) - main.count('\n') - 1
      assert macrono >= 0 and macrono < len(macros), f"error outside macro range: {e}"
      print(f"skipping {macros[macrono]}: {e}")
      del macros[macrono]
    except Exception as e: raise Exception("parsing failed") from e
  return main + '\n'.join(macros + epilog)
