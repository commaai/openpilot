import ctypes, itertools, re, functools, os
from tinygrad.helpers import unwrap
from tinygrad.runtime.autogen import libclang as clang # use REGEN=1 to regenerate libclang bindings

def unwrap_cursor(c: clang.CXCursor) -> clang.CXCursor:
  assert c != clang.clang_getNullCursor()
  return c

def children(c: clang.CXCursor) -> list[clang.CXCursor]:
  ret = []
  @clang.CXCursorVisitor
  def visitor(child, _0, _1):
    nonlocal ret
    ret.append(child)
    return clang.CXChildVisit_Continue
  clang.clang_visitChildren(c, visitor, None)
  return ret

def fields(t: clang.CXType) -> list[clang.CXCursor]:
  ret = []
  @clang.CXFieldVisitor
  def visitor(child, _):
    nonlocal ret
    ret.append(child)
    return clang.CXVisit_Continue
  clang.clang_Type_visitFields(t, visitor, None)
  return ret

# flattens anonymous fields
def all_fields(t, kind):
  for f in fields(t):
    if (clang.clang_Cursor_isAnonymousRecordDecl(clang.clang_getTypeDeclaration(clang.clang_getCursorType(f))) and
        clang.clang_getTypeDeclaration(clang.clang_getCursorType(f)).kind == kind):
      yield from all_fields(clang.clang_getCursorType(f), kind)
    else: yield f

def arguments(c: clang.CXCursor|clang.CXType):
  yield from ((clang.clang_Cursor_getArgument if isinstance(c, clang.CXCursor) else clang.clang_getArgType)(c, i)
              for i in range(clang.clang_Cursor_getNumArguments(c) if isinstance(c, clang.CXCursor) else clang.clang_getNumArgTypes(c)))

class Tokens:
  def __init__(self, c: clang.CXCursor):
    clang.clang_tokenize(tu:=clang.clang_Cursor_getTranslationUnit(c), clang.clang_getCursorExtent(c),
                         toks:=(ctypes.POINTER(clang.CXToken)()), cnt:=ctypes.c_uint32())
    self.tu, self.toks = tu, toks[:cnt.value]
    for t in self.toks: t._tu = tu

  def __getitem__(self, idx): return self.toks[idx]
  def __len__(self): return len(self.toks)

  def __del__(self):
    if self.toks: clang.clang_disposeTokens(self.tu, self.toks[0], len(self.toks))

def cxs(fn):
  @functools.wraps(fn)
  def wrap(*args, **kwargs) -> str:
    if ctypes.cast(clang.clang_getCString(cxs:=fn(*args, **kwargs)), ctypes.c_void_p).value is None: return ""
    ret = ctypes.string_at(clang.clang_getCString(cxs)).decode()
    clang.clang_disposeString(cxs)
    return ret
  return wrap

# TODO: caching this would be nice?
nm = cxs(lambda c: getattr(clang, f"clang_get{c.__class__.__name__[2:]}Spelling")(*([c._tu, c] if isinstance(c, clang.CXToken) else [c])))
def extent(c): return getattr(clang, f"clang_get{c.__class__.__name__[2:]}Extent")(*([c._tu, c] if isinstance(c, clang.CXToken) else [c]))
def loc(c): return getattr(clang, f"clang_get{c.__class__.__name__[2:]}Location")(*([c._tu, c] if isinstance(c, clang.CXToken) else [c]))
def gel(loc: clang.CXSourceLocation):
  clang.clang_getExpansionLocation(loc, file:=clang.CXFile(), line:=ctypes.c_uint32(), None, offset:=ctypes.c_uint32())
  return {"file":clang.clang_getFileName(file), "line":line.value, "offset":offset.value}
loc_file = cxs(lambda loc: gel(loc)['file'])
def loc_off(loc: clang.CXSourceLocation) -> int: return gel(loc)['offset']
def loc_line(loc: clang.CXSourceLocation) -> int: return gel(loc)['line']

def readext(f, fst, snd=None):
  with open(f, "r") as f: # reopening this every time is dumb...
    f.seek(start:=loc_off(clang.clang_getRangeStart(fst) if isinstance(fst, clang.CXSourceRange) else fst))
    return f.read(loc_off(clang.clang_getRangeEnd(fst) if isinstance(fst, clang.CXSourceRange) else snd)-start)
def attrs(c): return list(filter(lambda k: (v:=k.value) >= 400 and v < 500, map(lambda c: c.kind, children(c))))

def protocols(t): yield from (clang.clang_Type_getObjCProtocolDecl(t, i) for i in range(clang.clang_Type_getNumObjCProtocolRefs(t)))
def basetype(t): return clang.clang_Type_getObjCObjectBaseType(t)

base_rules = [(r'\s*\\\n\s*', ' '), (r'\s*\n\s*', ' '), (r'//.*', ''), (r'/\*.*?\*/', ''), (r'\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b', r'\1'),
              (r'\b0+(?=\d)', ''), (r'\s*&&\s*', r' and '), (r'\s*\|\|\s*', r' or '), (r'\s*!\s*', ' not '),
              (r'(struct|union|enum)\s*([a-zA-Z_][a-zA-Z0-9_]*\b)', r'\1_\2'),
              (r'\((unsigned )?(char|uint64_t)\)', ''), (r'^.*\d+:\d+.*$', ''), (r'^.*\w##\w.*$', '')]

uints = (clang.CXType_Char_U, clang.CXType_UChar, clang.CXType_UShort, clang.CXType_UInt, clang.CXType_ULong, clang.CXType_ULongLong)
ints = uints + (clang.CXType_Char_S, clang.CXType_Short, clang.CXType_Int, clang.CXType_ULong, clang.CXType_LongLong)
fns, specs = (clang.CXType_FunctionProto, clang.CXType_FunctionNoProto), (clang.CXCursor_ObjCSuperClassRef,) # this could include protocols
# https://clang.llvm.org/docs/AutomaticReferenceCounting.html#arc-method-families
arc_families = ['alloc', 'copy', 'mutableCopy', 'new']

def gen(name, dll, files, args=[], prolog=[], rules=[], epilog=[], recsym=False, errno=False, anon_names={}, types={}, parse_macros=True, paths=[]):
  macros, lines, anoncnt, types, objc = [], [], itertools.count().__next__, {k:(v,True) for k,v in types.items()}, False
  def tname(t, suggested_name=None, typedef=None) -> str:
    suggested_name = anon_names.get(f"{loc_file(loc(decl:=clang.clang_getTypeDeclaration(t)))}:{loc_line(loc(decl))}", suggested_name)
    nonlocal lines, types, anoncnt, objc
    tmap = {clang.CXType_Void:"None", clang.CXType_Char_U:"ctypes.c_ubyte", clang.CXType_UChar:"ctypes.c_ubyte", clang.CXType_Char_S:"ctypes.c_char",
            clang.CXType_SChar:"ctypes.c_byte",
            **{getattr(clang, f'CXType_{k}'):f"ctypes.c_{k.lower()}" for k in ["Bool", "WChar", "Float", "Double", "LongDouble"]},
            **{getattr(clang, f'CXType_{k}'):f"ctypes.c_{'u' if 'U' in k else ''}int{sz}" for sz,k in
               [(16, "UShort"), (16, "Short"), (32, "UInt"), (32, "Int"), (64, "ULong"), (64, "Long"), (64, "ULongLong"), (64, "LongLong")]}}

    if t.kind in tmap: return tmap[t.kind]
    if nm(t) in types and types[nm(t)][1]: return types[nm(t)][0]
    if ((f:=t).kind in fns) or (t.kind == clang.CXType_Pointer and (f:=clang.clang_getPointeeType(t)).kind in fns):
      return (f"ctypes.CFUNCTYPE({tname(clang.clang_getResultType(f))}" +
              ((', '+', '.join(map(tname, arguments(f)))) if f.kind==clang.CXType_FunctionProto else '') + ")")
    match t.kind:
      case clang.CXType_Pointer:
        return "ctypes.c_void_p" if (p:=clang.clang_getPointeeType(t)).kind==clang.CXType_Void else f"ctypes.POINTER({tname(p)})"
      case clang.CXType_ObjCObjectPointer: return tname(clang.clang_getPointeeType(t)) # TODO: this seems wrong
      case clang.CXType_Elaborated: return tname(clang.clang_Type_getNamedType(t), suggested_name)
      case clang.CXType_Typedef if nm(t) == nm(canon:=clang.clang_getCanonicalType(t)): return tname(canon)
      case clang.CXType_Typedef:
        defined, cnm = nm(canon:=clang.clang_getCanonicalType(t)) in types, tname(canon, typedef=nm(t).replace('::', '_'))
        types[nm(t)] = cnm if nm(t).startswith("__") else nm(t).replace('::', '_'), True
        # RECORDs need to handle typedefs specially to allow for self-reference
        if canon.kind != clang.CXType_Record or defined: lines.append(f"{nm(t).replace('::', '_')} = {cnm}")
        return types[nm(t)][0]
      case clang.CXType_Record:
        # TODO: packed unions
        # libclang does not use CXType_Elaborated for function parameters with type qualifiers (eg. void (*)(const struct foo))
        if (_nm:=re.sub(r"^const ", "", nm(t))) in types and types[_nm][1]: return types[_nm][0]
        # check for forward declaration
        if _nm in types: types[_nm] = (tnm:=types[_nm][0]), len(fields(t)) != 0
        else:
          if clang.clang_Cursor_isAnonymous(decl):
            types[_nm] = (tnm:=(suggested_name or (f"_anon{'struct' if decl.kind==clang.CXCursor_StructDecl else 'union'}{anoncnt()}")), True)
          else: types[_nm] = (tnm:=_nm.replace(' ', '_').replace('::', '_')), len(fields(t)) != 0
          lines.append(f"class {tnm}({'Struct' if decl.kind==clang.CXCursor_StructDecl else 'ctypes.Union'}): pass")
          if typedef: lines.append(f"{typedef} = {tnm}")
        if ((is_packed:=(clang.CXCursor_PackedAttr in attrs(decl)) or
            ((N:=clang.clang_Type_getAlignOf(t)) != max([clang.clang_Type_getAlignOf(clang.clang_getCursorType(f)) for f in fields(t)], default=N)))):
          if clang.clang_Type_getAlignOf(t) != 1:
            print(f"WARNING: ignoring alignment={clang.clang_Type_getAlignOf(t)} on {_nm}")
            is_packed = False
        acnt = itertools.count().__next__
        def is_anon(f): return clang.clang_Cursor_isAnonymousRecordDecl(clang.clang_getTypeDeclaration(clang.clang_getCursorType(f)))
        ll=["  ("+((fn:=f"'_{acnt()}'")+f", {tname(clang.clang_getCursorType(f), tnm+fn[1:-1])}" if is_anon(f) else f"'{nm(f)}', "+
            tname(clang.clang_getCursorType(f), f'{tnm}_{nm(f)}'))+(f',{clang.clang_getFieldDeclBitWidth(f)}' * clang.clang_Cursor_isBitField(f))+"),"
            for f in all_fields(t, decl.kind)]
        lines.extend(([f"{tnm}._anonymous_ = ["+", ".join(f"'_{i}'" for i in range(n))+"]"] if (n:=acnt()) else [])+
                     ([f"{tnm}._packed_ = True"] * is_packed)+([f"{tnm}._fields_ = [",*ll,"]"] if ll else []))
        return tnm
      case clang.CXType_Enum:
        # TODO: C++ and GNU C have forward declared enums
        if clang.clang_Cursor_isAnonymous(decl): types[nm(t)] = suggested_name or f"_anonenum{anoncnt()}", True
        else: types[nm(t)] = nm(t).replace(' ', '_').replace('::', '_'), True
        ety = clang.clang_getEnumDeclIntegerType(decl)
        def value(e): return (clang.clang_getEnumConstantDeclUnsignedValue if ety.kind in uints else clang.clang_getEnumConstantDeclValue)(e)
        lines.append(f"{types[nm(t)][0]} = CEnum({tname(ety)})\n" +
                     "\n".join(f"{nm(e)} = {types[nm(t)][0]}.define('{nm(e)}', {value(e)})" for e in children(decl)
                     if e.kind == clang.CXCursor_EnumConstantDecl) + "\n")
        return types[nm(t)][0]
      case clang.CXType_ConstantArray:
        return f"({tname(clang.clang_getArrayElementType(t),suggested_name.rstrip('s') if suggested_name else None)} * {clang.clang_getArraySize(t)})"
      case clang.CXType_IncompleteArray:
        return f"({tname(clang.clang_getArrayElementType(t), suggested_name.rstrip('s') if suggested_name else None)} * 0)"
      case clang.CXType_ObjCInterface:
        is_defn = bool([f.kind for f in children(decl) if f.kind in (clang.CXCursor_ObjCInstanceMethodDecl, clang.CXCursor_ObjCClassMethodDecl)])
        if (tnm:=nm(t)) not in types: lines.append(f"class {tnm}(objc.Spec): pass")
        types[tnm] = tnm, is_defn
        if is_defn:
          ims, cms = parse_objc_spec(decl, tnm, clang.CXCursor_ObjCInstanceMethodDecl), parse_objc_spec(decl, tnm, clang.CXCursor_ObjCClassMethodDecl)
          bases = [tname(clang.clang_getCursorType(b)) for b in children(decl) if b.kind in specs]
          lines.extend([*([f"{tnm}._bases_ = [{', '.join(bases)}]"] if bases else []),
                        *([f"{tnm}._methods_ = [", *ims, ']'] if ims else []), *([f"{tnm}._classmethods_ = [", *cms, ']'] if cms else [])])
        return tnm
      case clang.CXType_ObjCSel: return "objc.id_"
      case clang.CXType_ObjCId: return (objc:=True, "objc.id_")[1]
      case clang.CXType_ObjCObject:
        if basetype(t).kind != clang.CXType_ObjCId: raise NotImplementedError(f"generics unsupported: {nm(t)}")
        if len(ps:=[proto(p) for p in protocols(t)]) == 0:
          types[nm(t)] = "objc.id_", True
          return "objc.id_"
        if len(ps) == 1:
          types[nm(t)] = ps[0], True
          return ps[0]
        types[nm(t)] = (tnm:=f"_anondynamic{anoncnt()}"), True
        lines.append(f"class {tnm}({', '.join(ps)}): pass # {nm(t)}")
        return tnm
      case _: raise NotImplementedError(f"unsupported type {t.kind}")

  # parses an objc @interface or @protocol, returning a list of declerations that objc.Spec can parse, for the specified kind
  # NB: ivars are unsupported
  def parse_objc_spec(decl:clang.CXCursor, dnm:str, kind) -> list[str]:
    nonlocal lines, types
    ms = []
    for d in filter(lambda d: d.kind == kind, children(decl)):
      rollback = lines, types
      try: ms.append(f"  ('{nm(d)}', {repr('instancetype') if nm(rt:=clang.clang_getCursorResultType(d))=='instancetype' else tname(rt)}, " +
        f"[{', '.join('instancetype' if nm(a) == 'instancetype' else tname(clang.clang_getCursorType(a)) for a in arguments(d))}]" +
        (", True" * (clang.CXCursor_NSReturnsRetained in attrs(d) or (any(nm(d).startswith(s) for s in arc_families) and rt.kind!=clang.CXType_Void)))
                     + "),")
      except NotImplementedError as e:
        print(f"skipping {dnm}.{nm(d)}: {e}")
        lines, types = rollback
    return ms

  # libclang doesn't have a "type" for @protocol, so we have to do this here...
  def proto(decl):
    nonlocal lines, types
    if (dnm:=nm(decl)) in types and types[dnm][1]: return types[dnm][0]
    # check if this is a forward declaration
    is_defn = bool([f.kind for f in children(decl) if f.kind in (clang.CXCursor_ObjCInstanceMethodDecl, clang.CXCursor_ObjCClassMethodDecl)])
    if dnm not in types: lines.append(f"class {dnm}(objc.Spec): pass")
    types[dnm] = dnm, is_defn
    if is_defn:
      bases = [proto(b) for b in children(decl) if b.kind==clang.CXCursor_ObjCProtocolRef and nm(b) != nm(decl)]
      ims, cms = parse_objc_spec(decl, dnm, clang.CXCursor_ObjCInstanceMethodDecl), parse_objc_spec(decl, dnm, clang.CXCursor_ObjCClassMethodDecl)
      lines.extend([*([f"{dnm}._bases_ = [{', '.join(bases)}]"] if bases else []),
                    *([f"{dnm}._methods_ = [", *ims, "]"] if ims else []), *([f"{dnm}._classmethods_ = [", *cms, "]"] if cms else [])])
    return dnm

  for f in files:
    aa = ctypes.cast((ctypes.c_char_p * len(args))(*[x.encode() for x in args]), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))) if len(args) else None
    tu = clang.clang_parseTranslationUnit(idx:=clang.clang_createIndex(False, 0), os.fspath(f).encode(), aa, len(args), None, 0,
                                          clang.CXTranslationUnit_DetailedPreprocessingRecord)
    q = list(children(unwrap_cursor(clang.clang_getTranslationUnitCursor(tu))))[::-1]
    while q:
      c = q.pop()
      if loc_file(loc(c)) != str(f) and (not recsym or c.kind not in (clang.CXCursor_FunctionDecl,)): continue
      rollback = lines, types
      try:
        match c.kind:
          case clang.CXCursor_FunctionDecl if clang.clang_getCursorLinkage(c) == clang.CXLinkage_External and dll:
            # TODO: we could support name-mangling
            lines.append(f"try: ({nm(c)}:=dll.{nm(c)}).restype, {nm(c)}.argtypes = {tname(clang.clang_getCursorResultType(c))}, "
                         f"[{', '.join(tname(clang.clang_getCursorType(arg)) for arg in arguments(c))}]\nexcept AttributeError: pass\n")
            if clang.CXCursor_NSReturnsRetained in attrs(c): lines.append(f"{nm(c)} = objc.returns_retained({nm(c)})")
          case (clang.CXCursor_StructDecl | clang.CXCursor_UnionDecl | clang.CXCursor_TypedefDecl | clang.CXCursor_EnumDecl
                | clang.CXCursor_ObjCInterfaceDecl): tname(clang.clang_getCursorType(c))
          case clang.CXCursor_MacroDefinition if parse_macros and len(toks:=Tokens(c)) > 1:
            if nm(toks[1])=='(' and clang.clang_equalLocations(clang.clang_getRangeEnd(extent(toks[0])), clang.clang_getRangeStart(extent(toks[1]))):
              it = iter(toks[1:])
              _args = [nm(t) for t in itertools.takewhile(lambda t:nm(t)!=')', it) if clang.clang_getTokenKind(t) == clang.CXToken_Identifier]
              if len(body:=list(it)) == 0: continue
              macros += [f"{nm(c)} = lambda{' ' * bool(_args)}{','.join(_args)}: {readext(f,loc(body[0]),clang.clang_getRangeEnd(extent(toks[-1])))}"]
            else: macros += [f"{nm(c)} = {readext(f, loc(toks[1]), clang.clang_getRangeEnd(extent(toks[-1])))}"]
          case clang.CXCursor_VarDecl if clang.clang_getCursorLinkage(c) == clang.CXLinkage_Internal:
            ty = clang.clang_getCursorType(c)
            if (ty.kind == clang.CXType_ConstantArray and clang.clang_getCanonicalType(clang.clang_getArrayElementType(ty)).kind in ints and
                (init:=children(c)[-1]).kind == clang.CXCursor_InitListExpr
                and all(re.match(r"\[.*\].*=", readext(f, extent(c))) for c in children(init))):
              cs = children(init)
              macros += [f"{nm(c)} = {{{','.join(f'{readext(f, extent(next(it:=iter(children(c)))))}:{readext(f, extent(next(it)))}' for c in cs)}}}"]
            elif clang.clang_getCanonicalType(ty).kind in ints: macros += [f"{nm(c)} = {readext(f, extent(children(c)[-1]))}"]
            else: macros += [f"{nm(c)} = {tname(ty)}({readext(f, extent(children(c)[-1]))})"]
          case clang.CXCursor_VarDecl if clang.clang_getCursorLinkage(c) == clang.CXLinkage_External and dll:
            lines.append(f"try: {nm(c)} = {tname(clang.clang_getCursorType(c))}.in_dll(dll, '{nm(c)}')\nexcept (ValueError,AttributeError): pass")
          case clang.CXCursor_ObjCProtocolDecl: proto(c)
          case clang.CXCursor_Namespace | clang.CXCursor_LinkageSpec: q.extend(list(children(c))[::-1])
      except NotImplementedError as e:
        print(f"skipping {nm(c)}: {e}")
        lines, types = rollback
    clang.clang_disposeTranslationUnit(tu)
    clang.clang_disposeIndex(idx)
  main = '\n'.join(["# mypy: ignore-errors", "import ctypes", "from tinygrad.runtime.support.c import DLL, Struct, CEnum, _IO, _IOW, _IOR, _IOWR",
                    *prolog, *(["from tinygrad.runtime.support import objc"]*objc),
                    *([f"dll = DLL('{name}', {dll}{f', {paths}'*bool(paths)}{', use_errno=True'*errno})"] if dll else []), *lines]) + '\n'
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
