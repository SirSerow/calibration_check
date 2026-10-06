"""Unpublished draft only. Upload images, confirm each upload, read back draft."""
import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from urllib.parse import urlparse
from .io import read,write,digest,sha256

from .environment import load_env
load_env()

ENDPOINT='https://gql-beta.hashnode.com'


def graphql(query,variables):
    token=os.environ.get('HASHNODE_PAT')
    if not token:
        raise RuntimeError('HASHNODE_PAT is unset; export it in the execution environment')
    request=urllib.request.Request(ENDPOINT,data=json.dumps(dict(query=query,variables=variables)).encode(),
                                  headers={'Authorization':'Bearer '+token,'Content-Type':'application/json','x-hashnode-client':'gql-skill','User-Agent':'model-calibration/0.1'})
    try:
        with urllib.request.urlopen(request,timeout=60) as response:
            body=json.load(response)
    except urllib.error.HTTPError as error:
        message=error.read().decode()[:1000].replace(token,'[REDACTED]')
        raise RuntimeError(f'Hashnode HTTP {error.code}: {message}') from None
    if body.get('errors'):
        # Do not retry any mutation, particularly the documented Pro gate.
        errors=[dict(message=e['message'],code=e.get('extensions',{}).get('code')) for e in body['errors']]
        raise RuntimeError(json.dumps(errors))
    return body['data']


def create_draft(run,publication,methodology_url):
    if not os.getenv('HASHNODE_PAT'):
        raise RuntimeError('HASHNODE_PAT is unset; export it in the execution environment')
    run=Path(run)
    manifest=read(run/'manifest.json')
    if manifest['synthetic']:
        raise ValueError('Synthetic results cannot be uploaded as an evidence-backed article')
    validation=read(run/'article_validation.json')
    if validation['summary_hash']!=digest(read(run/'summary.json')) or validation['correction_hash']!=digest(read(run/'correction_results.json')):
        raise ValueError('Article results changed after validation')
    article=Path('article/article.md')
    if validation['article_sha256']!=sha256(article):
        raise ValueError('Article changed after validation')
    if not methodology_url.startswith('https://'):
        raise ValueError('A public HTTPS methodology link is required for the online draft')
    publication=publication or os.getenv('HASHNODE_PUBLICATION_ID') or os.getenv('HASHNODE_PUBLICATION_URL')
    if not publication:
        raise RuntimeError('Target Hashnode publication URL or ID is required')
    if '://' in publication:
        host=urlparse(publication).netloc
        publication=graphql('query ($host:String!) { publication(host:$host) { id } }',{'host':host})['publication']['id']
    state_path=run/'hashnode_draft.json'
    state=read(state_path) if state_path.exists() else dict(publication_id=publication,images={})
    target=Path('results/hashnode-target.json')
    if not state_path.exists() and target.exists():
        existing=read(target)
        if existing['publication_id']==publication and existing.get('existing_draft_empty'):
            state['draft_id']=existing['draft_id']
    if state['publication_id']!=publication:
        raise ValueError('Draft belongs to a different publication')
    content=article.read_text().replace('../README.md',methodology_url)
    for name in ['calibration_comparison','correction','quality_vs_calibration']:
        path=Path('results/figures')/(name+'.png')
        fingerprint=sha256(path)
        image_state=state['images'].get(name)
        if image_state and image_state['sha256']!=fingerprint:
            raise ValueError('Uploaded figure changed; use a new draft state')
        if not image_state:
            upload=graphql('mutation ($input:CreateImageUploadInput!) { createImageUploadURL(input:$input) { presignedPut { url cdnUrl key } } }',
                           {'input':{'contentType':'image/png'}})['createImageUploadURL']['presignedPut']
            if path.stat().st_size>8*1024*1024:
                raise ValueError('PNG exceeds Hashnode 8 MB image limit')
            req=urllib.request.Request(upload['url'],data=path.read_bytes(),method='PUT',headers={'Content-Type':'image/png'})
            with urllib.request.urlopen(req,timeout=60) as response:
                response.read()
            confirmed=graphql('mutation ($input:ConfirmImageUploadInput!) { confirmImageUpload(input:$input) { ok cdnUrl } }',
                              {'input':{'key':upload['key']}})['confirmImageUpload']
            if not confirmed['ok']:
                raise RuntimeError('Hashnode image confirmation rejected the image')
            image_state=dict(url=confirmed['cdnUrl'],sha256=fingerprint)
            state['images'][name]=image_state
            write(state_path,state)
        content=content.replace(f'../results/figures/{name}.png',image_state['url'])
    title=content.splitlines()[0].removeprefix('# ')
    if '{{' in content or '../' in content:
        raise ValueError('Online article contains unresolved links or placeholders')
    if state.get('draft_id'):
        graphql('mutation ($input:UpdateDraftInput!) { updateDraft(input:$input) { draft { id } } }',
                {'input':{'draftId':state['draft_id'],'title':title,'contentMarkdown':content}})
    else:
        state['draft_id']=graphql('mutation ($input:CreateDraftInput!) { createDraft(input:$input) { draft { id } } }',
                                  {'input':{'publicationId':publication,'title':title,'contentMarkdown':content}})['createDraft']['draft']['id']
        state['verified']=False
        write(state_path,state) # Persist ID before attempting read-back.
    result=graphql('query ($id:ObjectId!) { draft(id:$id) { id title content { markdown } } }',{'id':state['draft_id']})['draft']
    if result['title']!=title or result['content']['markdown']!=content:
        raise RuntimeError('Hashnode draft content did not match submitted content')
    state.update(verified=True,content_hash=digest(content))
    write(state_path,state)
    return state
